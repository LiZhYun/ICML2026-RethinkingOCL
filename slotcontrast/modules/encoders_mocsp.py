"""MOCSP (D3) encoder extensions for Masked Object-Conditioned Self-Prediction.

This module provides ``MaskingFrameEncoder``, a subclass of
``FrameEncoder`` that injects a learnable mask token into a random
fraction of patch-level projected features during training. The mask
lives at the slot-input granularity (post ``output_transform``), so the
downstream slot attention consumes masked features while the raw
``backbone_features`` target remains intact for reconstruction losses.

Design constraints:
 - MUST NOT modify ``FrameEncoder`` in place (CLAUDE.md rule).
 - MUST expose ``input_mask`` in the encoder output dict so
   ``MOCSPLoss`` can locate the held-out patches.
 - Validation runs without masking by default (``mask_on_eval=False``)
   so metrics are evaluated on the true feature manifold.
"""
from typing import Dict, Optional

import torch
from torch import nn

from slotcontrast.modules import utils as modules_utils
from slotcontrast.modules.encoders import FrameEncoder


class MaskingFrameEncoder(FrameEncoder):
    """FrameEncoder with MAE-style patch masking for D3 MOCSP.

    After the parent's feature pipeline produces ``features`` (post
    ``output_transform``), a Bernoulli mask of ratio ``mask_ratio`` is
    drawn over the patch dimension and the masked positions are replaced
    with a learnable ``mask_token``. ``backbone_features`` (the frozen
    target signal) are never masked — only the slot-input features.

    The mask is exposed as ``input_mask: [B, P]`` in the output dict so
    downstream losses (MOCSPLoss) can locate held-out patches.

    Args:
        backbone / pos_embed / output_transform / ...: same as FrameEncoder.
            Accepts either pre-built ``nn.Module`` instances *or* config
            dicts; dicts are built via ``utils.build_module`` to keep
            compatibility with the fallback registry path.
        mask_ratio: fraction of patches to replace with the mask token.
        mask_on_eval: if True, also apply masking during eval (for
            debugging / diagnostics); default False.
        mask_token_dim: output feature dim; inferred from
            ``output_transform.outp_dim`` when available.
    """

    def __init__(
        self,
        backbone,
        pos_embed=None,
        output_transform=None,
        mask_ratio: float = 0.5,
        mask_on_eval: bool = False,
        mask_token_dim: Optional[int] = None,
        mask_type: str = "learned_token",
        noise_std: float = 0.02,
        main_features_key: str = "vit_block12",
        use_pos_embed: bool = True,
        **kwargs,
    ):
        # Build sub-modules if they are still config dicts (support the
        # fallback registry path where the build function does not do the
        # sub-module construction for us).
        if backbone is not None and not isinstance(backbone, nn.Module):
            backbone = modules_utils.build_module(backbone, default_group="encoders")
        if pos_embed is not None and not isinstance(pos_embed, nn.Module):
            pos_embed = modules_utils.build_module(pos_embed) if use_pos_embed else None
        if output_transform is not None and not isinstance(output_transform, nn.Module):
            output_transform = modules_utils.build_module(output_transform)

        super().__init__(
            backbone=backbone,
            pos_embed=pos_embed,
            output_transform=output_transform,
            main_features_key=main_features_key,
            **kwargs,
        )

        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"MaskingFrameEncoder.mask_ratio must be in [0, 1], got {mask_ratio}"
            )
        self.mask_ratio = float(mask_ratio)
        self.mask_on_eval = bool(mask_on_eval)
        valid_mask_types = ("learned_token", "zero", "gaussian_noise", "dropout")
        if mask_type not in valid_mask_types:
            raise ValueError(
                f"MaskingFrameEncoder.mask_type must be one of {valid_mask_types}, "
                f"got {mask_type!r}"
            )
        self.mask_type = mask_type
        self.noise_std = float(noise_std)

        if mask_token_dim is None:
            # Try exposed attribute first.
            candidate = getattr(output_transform, "outp_dim", None)
            if candidate is None:
                candidate = getattr(output_transform, "output_dim", None)
            # Fall back to inspecting the last nn.Linear in a Sequential MLP.
            if candidate is None and output_transform is not None:
                inner = getattr(output_transform, "layers", None)
                if isinstance(inner, nn.Sequential):
                    for mod in reversed(list(inner)):
                        if isinstance(mod, nn.Linear):
                            candidate = mod.out_features
                            break
            if candidate is None:
                raise ValueError(
                    "MaskingFrameEncoder.mask_token_dim must be specified when "
                    "it cannot be inferred from `output_transform`. Set "
                    "`mask_token_dim: <SLOT_DIM>` in the encoder config."
                )
            mask_token_dim = int(candidate)

        # mask_token initialised small so the early-training path is close
        # to the unmasked features; LoRA can stabilise before the masks
        # start carrying strong signal. Allocated for all mask_types so
        # checkpoint shapes match when switching modes; only used when
        # mask_type == "learned_token".
        self.mask_token = nn.Parameter(torch.randn(1, 1, mask_token_dim) * 0.02)

    def _maybe_apply_mask(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply patch masking to ``features`` when training.

        Args:
            features: ``[B, P, D]`` patch-level features to be consumed by
                slot attention / initializer.

        Returns:
            dict with masked ``features`` and the ``input_mask`` (``[B, P]``
            float tensor, 1.0 = masked / held-out).
        """
        B, P, D = features.shape
        should_mask = (self.training or self.mask_on_eval) and self.mask_ratio > 0.0
        if not should_mask:
            # Return a zero mask so downstream losses can safely index it.
            input_mask = features.new_zeros((B, P))
            return {"features": features, "input_mask": input_mask}

        # Per-sample random masking (Bernoulli). Independent draws per
        # (batch, patch) — the stochastic sampling is the per-step signal
        # that the slot pipeline must predict held-out patch features.
        input_mask = (torch.rand(B, P, device=features.device) < self.mask_ratio).float()
        mask_expanded = input_mask.unsqueeze(-1).bool()

        if self.mask_type == "learned_token":
            fill = self.mask_token.to(features.dtype).expand(B, P, D)
        elif self.mask_type == "zero":
            fill = torch.zeros_like(features)
        elif self.mask_type == "gaussian_noise":
            fill = torch.randn_like(features) * self.noise_std
        elif self.mask_type == "dropout":
            # Standard-dropout style: scale surviving features by 1/(1-p)
            # so the expectation matches unmasked training. Masked positions
            # become zero; surviving (1-m) positions get multiplied by
            # 1/(1-p).
            survive_scale = 1.0 / max(1.0 - self.mask_ratio, 1e-6)
            masked_features = features * (1.0 - input_mask.unsqueeze(-1)) * survive_scale
            return {"features": masked_features, "input_mask": input_mask}
        else:
            raise RuntimeError(f"Unhandled mask_type {self.mask_type!r}")

        masked_features = torch.where(mask_expanded, fill, features)
        return {"features": masked_features, "input_mask": input_mask}

    def forward(
        self,
        images: torch.Tensor,
        camera_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(images, camera_data=camera_data)
        patched = self._maybe_apply_mask(out["features"])
        out = {**out, **patched}
        return out
