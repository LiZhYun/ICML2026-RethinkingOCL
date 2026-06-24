# ============================================================================
# TPAMI extension — active vs quarantined losses (Round-32 reviewer audit).
#
# ON the active rescue path (used by configs/slotcontrast/app/*rescue15k.yaml):
#   * MSELoss               (loss_featrec, target = teacher backbone features)
#   * Slot_Slot_Contrastive_Loss  (loss_ss)
# QUARANTINED — not invoked by any rescue/app config in the v2 grid or any
# follow-up grid (Phase A, Phase B, F.4, YT-VIS LR-sweep):
#   * TubeGramLoss          (parked 2026-04; null finding per REVIEWER_MEMORY)
#   * GlobalGramLoss        (parked; alternative anchor formulation)
#   * MOCSPLoss             (D3 prototype; ICML 2026 oral pivot artifact)
#   * χ-Boost / replay / inpainter / occlusion / saliency-temporal losses
#     (assorted prototypes carried over from prior rounds; not on TPAMI path)
# Each quarantined loss's `forward` is a no-op zero return when the loss is
# gated off (e.g. Phase 1 ramp factor = 0 or no valid anchors / negatives).
# These zero returns are documented degenerate-case handlers, not silent
# fallbacks: the active TPAMI rescue path never instantiates these losses.
# ============================================================================
import math
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
from torch import nn

from slotcontrast import modules, utils
# Authoritative event-category → index mapping (shared with
# ``modules.replay`` and ``modules.networks.OpenSetIdentityHead``). Imported
# directly (not via ``slotcontrast.modules``) so this file does not force
# the rest of modules/ to load before losses/.
from slotcontrast.modules.replay import EVENT_TO_INDEX as _SHARED_EVENT_TO_INDEX


@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)


class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)


class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.batch_contrast = batch_contrast
        self.mask_key = mask_key

    def forward(self, slots, _, existence_mask=None):
        """
        Args:
            slots: [B, T, K, D] slot representations
            _: unused target
            existence_mask: [B, T, K] or [B, K] - 1.0 for valid slots, 0.0 for empty
        """
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        
        # Handle existence mask
        if existence_mask is not None:
            # Expand [B, K] to [B, T, K] if needed
            if existence_mask.ndim == 2 and slots.ndim == 4:
                existence_mask = existence_mask.unsqueeze(1).expand(-1, slots.shape[1], -1)
        
        if self.batch_contrast:
            slots_list = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots_list, dim=-2)  # 1xTxK*BxD
            if existence_mask is not None:
                mask_list = existence_mask.split(1)  # [1xTxK]
                existence_mask = torch.cat(mask_list, dim=-1)  # 1xTxK*B
        
        s1 = slots[:, :-1, :, :]  # [B, T-1, S, D]
        s2 = slots[:, 1:, :, :]   # [B, T-1, S, D]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature  # [B, T-1, S, S]
        B, T, S, _ = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        
        # Compute per-element loss
        loss_per_slot = self.criterion(ss, target)  # [B*T, S, S] -> [B*T, S]
        
        if existence_mask is not None:
            # Build valid pair mask: both slots at t and t+1 must exist
            m1 = existence_mask[:, :-1, :]  # [B, T-1, S]
            m2 = existence_mask[:, 1:, :]   # [B, T-1, S]
            # For slot i, it's valid if slot i exists at both t and t+1
            # The loss matrix ss[i,j] compares slot i at t with slot j at t+1
            # For diagonal (target), we need both slot i to exist at t and t+1
            valid_mask = (m1 * m2).reshape(B * T, S)  # [B*T, S]
            
            # Mask invalid slots (average only over valid)
            valid_count = valid_mask.sum()
            if valid_count > 0:
                loss = (loss_per_slot * valid_mask).sum() / valid_count
            else:
                loss = loss_per_slot.mean()  # Fallback if no valid pairs
        else:
            loss = loss_per_slot.mean()
        
        return loss


class TubeGramLoss(Loss):
    """Object-conditioned covariance anchoring for stable backbone finetuning.

    Computes per-object centered, trace-normalized covariance matrices on
    backbone patch features (grouped by decoder slot masks) and penalises
    divergence from an EMA teacher's covariance structure.  This preserves
    each object's dense patch manifold while allowing inter-object means to
    separate — exactly the property needed for stable end-to-end finetuning.
    """

    def __init__(
        self,
        pred_key: str = "encoder.backbone_features",
        target_key: str = "ema_backbone_features",
        confidence_threshold: float = 0.05,
        eps: float = 1e-6,
        mask_key: str = "existence_mask",
        centered: bool = True,
        trace_normalize: bool = True,
        detach_masks: bool = True,
        soft_gate: bool = True,
        soft_gate_temperature: float = 0.05,
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key,
            target_key,
            video_inputs=True,
            patch_inputs=False,
            keep_input_dim=True,
            **kwargs,
        )
        self.confidence_threshold = confidence_threshold
        self.eps = eps
        self.mask_key = mask_key
        self.centered = centered
        self.trace_normalize = trace_normalize
        self.detach_masks = detach_masks
        self.soft_gate = soft_gate
        self.soft_gate_temperature = soft_gate_temperature
        # Round 2 A1: consume TEACHER masks, not student (proposal §Method:
        # "Build tubes from EMA teacher, not student"). Student masks drift
        # with LoRA so anchoring to them is chicken-and-egg. Teacher masks
        # are emitted by the snapshot pipeline in `models.forward` on
        # training steps after `_backbone_unfreeze_step`.
        self.aux_keys = ["teacher_decoder_masks_soft"]
        # Diagnostics pulled by models.training_step for W&B logging.
        self._last_conf_mean: Optional[torch.Tensor] = None
        self._last_gate_mean: Optional[torch.Tensor] = None
        self._last_gate_active_frac: Optional[torch.Tensor] = None

    def get_target(self, inputs, outputs):
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            return None
        return target.detach()

    def _object_covariance(
        self, features: torch.Tensor, masks: torch.Tensor,
    ) -> torch.Tensor:
        """Centered, trace-normalised per-object covariance.

        Args:
            features: [B, T, P, D] L2-normalised backbone features.
            masks:    [B, T, K, P] soft slot assignments (sum-to-1 over K).

        Returns:
            covs: [B, K, D, D] trace-normalised centered covariance per object.
        """
        B, T, P, D = features.shape
        K = masks.shape[2]

        weights = masks.transpose(-1, -2)                   # [B, T, P, K]
        weights = weights.reshape(B, T * P, K)              # [B, TP, K]
        feats = features.reshape(B, T * P, D)               # [B, TP, D]

        w_sum = weights.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [B, 1, K]

        if self.centered:
            mu = torch.einsum("bpk,bpd->bkd", weights, feats)  # [B, K, D]
            mu = mu / w_sum.transpose(-1, -2)                   # [B, K, D]
            residual = feats.unsqueeze(2) - mu.unsqueeze(1)     # [B, TP, K, D]
        else:
            residual = feats.unsqueeze(2).expand(-1, -1, K, -1)

        w_expanded = weights.unsqueeze(-1)                  # [B, TP, K, 1]
        cov = torch.einsum(
            "bpki,bpkj->bkij",
            residual * w_expanded,
            residual,
        )                                                    # [B, K, D, D]
        cov = cov / w_sum.transpose(-1, -2).unsqueeze(-1).clamp(min=self.eps)

        if self.trace_normalize:
            trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
            cov = cov / trace.clamp(min=self.eps)

        return cov

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        existence_mask=None,
        teacher_decoder_masks_soft=None,
        **kwargs,
    ) -> torch.Tensor:
        # Round 2 A1: `teacher_decoder_masks_soft` is emitted only on
        # training steps after `backbone_unfreeze_step` (Phase 2). On
        # Phase 1 steps and on validation steps the teacher pipeline is
        # not run and this kwarg is None — return zero loss. The ramp
        # factor in `compute_loss` is also 0 during Phase 1 so this is
        # double protection.
        if teacher_features is None or teacher_decoder_masks_soft is None:
            return student_features.new_zeros(())

        # Teacher masks are stop-gradient (frozen pipeline). `detach_masks`
        # keeps us correct if a future variant pipes gradient-carrying masks.
        masks = (
            teacher_decoder_masks_soft.detach()
            if self.detach_masks
            else teacher_decoder_masks_soft
        )
        B, T, K, P = masks.shape

        # Margin-over-uniform gate (Round 1 fix — Critical #1).
        # Hard thresholding silently zeroes the loss before slot masks sharpen.
        # Soft sigmoid weighting is continuous in the margin, so gradient flows
        # proportionally to confidence even when no slot crosses the threshold.
        conf = masks.max(dim=-1).values.mean(dim=1)  # [B, K]
        margin = conf - (1.0 / K)
        if self.soft_gate:
            conf_mask = torch.sigmoid(
                (margin - self.confidence_threshold) / self.soft_gate_temperature
            )
        else:
            conf_mask = (margin > self.confidence_threshold).float()

        sf = torch.nn.functional.normalize(student_features, dim=-1)
        tf = torch.nn.functional.normalize(teacher_features, dim=-1)

        cov_s = self._object_covariance(sf, masks)  # [B, K, D, D]
        cov_t = self._object_covariance(tf, masks)  # [B, K, D, D]

        diff = (cov_s - cov_t.detach()).pow(2).sum(dim=(-2, -1))  # [B, K]

        # Denominator = count of *possible* slots (optionally filtered by
        # existence_mask) — NOT the weighted sum of conf_mask. Using the
        # weighted sum as the denominator would cancel the gate entirely
        # (if conf_mask = c everywhere, loss becomes the plain mean of diff).
        # By using a fixed count we preserve magnitude scaling: low-confidence
        # slots contribute proportionally less gradient. (Round 1 review fix.)
        base = torch.ones_like(conf_mask)
        if existence_mask is not None and existence_mask.ndim == 3:
            slot_valid = existence_mask.any(dim=1).float()
            conf_mask = conf_mask * slot_valid
            base = base * slot_valid

        denom = base.sum().clamp(min=1.0)
        loss = (diff * conf_mask).sum() / denom

        with torch.no_grad():
            self._last_conf_mean = conf.detach().mean()
            self._last_gate_mean = conf_mask.detach().mean()
            self._last_gate_active_frac = (conf_mask.detach() > 0.5).float().mean()

        return loss


class GlobalGramLoss(Loss):
    """Image-level covariance anchoring (DINOv3-style, no object conditioning).

    Computes a single centered, trace-normalized covariance matrix over ALL
    backbone patches and penalises divergence from the EMA teacher.  This is
    the ablation baseline for TubeGramLoss — it anchors the entire feature
    manifold rather than per-object sub-manifolds.
    """

    def __init__(
        self,
        pred_key: str = "encoder.backbone_features",
        target_key: str = "ema_backbone_features",
        eps: float = 1e-6,
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key,
            target_key,
            video_inputs=True,
            patch_inputs=False,
            keep_input_dim=True,
            **kwargs,
        )
        self.eps = eps

    def get_target(self, inputs, outputs):
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            return None
        return target.detach()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if teacher_features is None:
            return student_features.new_zeros(())

        B, T, P, D = student_features.shape
        sf = torch.nn.functional.normalize(
            student_features.reshape(B, T * P, D), dim=-1,
        )
        tf = torch.nn.functional.normalize(
            teacher_features.reshape(B, T * P, D), dim=-1,
        )

        mu_s = sf.mean(dim=1, keepdim=True)
        mu_t = tf.mean(dim=1, keepdim=True)

        cov_s = torch.bmm((sf - mu_s).transpose(1, 2), sf - mu_s) / (T * P)
        cov_t = torch.bmm((tf - mu_t).transpose(1, 2), tf - mu_t) / (T * P)

        trace_s = torch.diagonal(cov_s, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        trace_t = torch.diagonal(cov_t, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        cov_s = cov_s / trace_s.clamp(min=self.eps)
        cov_t = cov_t / trace_t.clamp(min=self.eps)

        loss = (cov_s - cov_t.detach()).pow(2).sum(dim=(-2, -1)).mean()
        return loss


class ChiBoostLoss(Loss):
    """Inter-object repulsion on raw backbone features.

    Directly increases inter-object feature separability ``Delta`` in
    chi = Delta^2 / (sigma^2 * log K).  Combined with TubeGramLoss (which
    preserves within-object covariance sigma^2), this pair forms an
    explicit chi-maximising objective rather than relying on the indirect
    chain through slot-slot contrastive.

    Per slot we compute the confidence-weighted mean feature mu_k of the
    student backbone.  We L2-normalise these means and minimise the mean
    off-diagonal cosine similarity across valid slot pairs — equivalent
    to maximising sum_{i != j} ||mu_i - mu_j||^2 on the unit sphere.
    """

    def __init__(
        self,
        pred_key: str = "encoder.backbone_features",
        target_key: Optional[str] = None,
        confidence_threshold: float = 0.05,
        eps: float = 1e-6,
        mask_key: str = "existence_mask",
        detach_masks: bool = True,
        soft_gate: bool = True,
        soft_gate_temperature: float = 0.05,
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        # target_key is unused; we just need student features + masks.
        super().__init__(
            pred_key,
            target_key or pred_key,
            video_inputs=True,
            patch_inputs=False,
            keep_input_dim=True,
            **kwargs,
        )
        self.confidence_threshold = confidence_threshold
        self.eps = eps
        self.mask_key = mask_key
        self.detach_masks = detach_masks
        self.soft_gate = soft_gate
        self.soft_gate_temperature = soft_gate_temperature
        # Round 2 A1: consume TEACHER masks (snapshot pipeline), not student.
        # Prevents chicken-and-egg where ChiBoost pushes student means apart
        # using student-mask groupings that themselves drift with LoRA.
        self.aux_keys = ["teacher_decoder_masks_soft"]
        self._last_conf_mean: Optional[torch.Tensor] = None
        self._last_gate_mean: Optional[torch.Tensor] = None
        self._last_gate_active_frac: Optional[torch.Tensor] = None

    def get_target(self, inputs, outputs):
        return None

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features=None,
        existence_mask=None,
        teacher_decoder_masks_soft=None,
        **kwargs,
    ) -> torch.Tensor:
        # Round 2 A1: teacher masks are only emitted on Phase-2 training
        # steps — return zero on Phase 1 and on validation. Ramp factor
        # in `compute_loss` is 0 during Phase 1 too, so this is belt-and-
        # braces.
        if teacher_decoder_masks_soft is None:
            return student_features.new_zeros(())

        masks = (
            teacher_decoder_masks_soft.detach()
            if self.detach_masks
            else teacher_decoder_masks_soft
        )
        B, T, K, P = masks.shape
        _, _, _, D = student_features.shape

        weights = masks.transpose(-1, -2).reshape(B, T * P, K)   # [B, TP, K]
        feats = student_features.reshape(B, T * P, D)            # [B, TP, D]

        w_sum = weights.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [B, 1, K]
        mu = torch.einsum("bpk,bpd->bkd", weights, feats)             # [B, K, D]
        mu = mu / w_sum.transpose(-1, -2)

        # Margin-over-uniform gate — soft sigmoid form (Round 1 fix, Critical #1).
        conf = masks.max(dim=-1).values.mean(dim=1)                   # [B, K]
        margin = conf - (1.0 / K)
        if self.soft_gate:
            valid = torch.sigmoid(
                (margin - self.confidence_threshold) / self.soft_gate_temperature
            )
        else:
            valid = (margin > self.confidence_threshold).float()
        mu_norm = torch.nn.functional.normalize(mu, dim=-1, eps=self.eps)  # [B, K, D]
        sim = torch.bmm(mu_norm, mu_norm.transpose(1, 2))                  # [B, K, K]

        eye = torch.eye(K, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        base_pair = 1.0 - eye  # [1, K, K]

        if existence_mask is not None and existence_mask.ndim == 3:
            slot_valid = existence_mask.any(dim=1).float()
            valid = valid * slot_valid
            base_pair = slot_valid.unsqueeze(1) * slot_valid.unsqueeze(2) * base_pair

        pair_weight = valid.unsqueeze(1) * valid.unsqueeze(2) * (1.0 - eye)

        # Denominator = count of possible (non-diagonal, existence-valid) pairs;
        # using `pair_weight.sum()` as the denominator cancels the gate. See
        # TubeGramLoss comment. (Round 1 review fix.)
        denom = base_pair.sum().clamp(min=1.0)
        loss = (sim * pair_weight).sum() / denom

        with torch.no_grad():
            self._last_conf_mean = conf.detach().mean()
            self._last_gate_mean = valid.detach().mean()
            self._last_gate_active_frac = (valid.detach() > 0.5).float().mean()

        return loss


class GapContrastLoss(Loss):
    """Slot-slot contrastive loss over multiple temporal gaps with continuity masking.

    Extends the adjacent-only `Slot_Slot_Contrastive_Loss` by drawing positive pairs
    at gaps Delta in `gaps` (e.g., {1, 2, 4} frames apart), each pair masked by a
    DETACHED binary `continuity_mask` indicating same-identity slot persistence
    across the gap. Negatives are batch-contrastive: other slots in the paired
    frame (and, when `batch_contrast=True`, slots from other videos in the batch).

    The `continuity_mask` MUST be supplied by the caller as a detached binary
    tensor (see round-23 correction C6). It is never backpropagated through.
    Typical sources:
      - GT track-id derived mask (Table B).
      - Frozen heuristic (Table A), e.g., cosine-similarity thresholded across
        adjacent frames, computed in a `torch.no_grad()` block.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        gaps: Tuple[int, ...] = (1, 2, 4),
        gap_weights: Optional[Tuple[float, ...]] = None,
        batch_contrast: bool = True,
        mask_key: str = "continuity_mask",
        require_all_gaps_valid: bool = False,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        if len(gaps) == 0:
            raise ValueError("`gaps` must contain at least one positive integer gap.")
        if any(g <= 0 for g in gaps):
            raise ValueError(f"All entries in `gaps` must be > 0, got {gaps}.")
        self.gaps = tuple(int(g) for g in gaps)
        if gap_weights is None:
            gap_weights = tuple(1.0 / len(self.gaps) for _ in self.gaps)
        else:
            if len(gap_weights) != len(self.gaps):
                raise ValueError(
                    f"`gap_weights` length {len(gap_weights)} must match `gaps` length "
                    f"{len(self.gaps)}."
                )
            gap_weights = tuple(float(w) for w in gap_weights)
        self.gap_weights = gap_weights
        self.batch_contrast = batch_contrast
        self.mask_key = mask_key
        self.require_all_gaps_valid = require_all_gaps_valid

    def forward(self, pred, target=None, continuity_mask=None, **kwargs):
        """
        Args:
            pred: [B, T, N, D] slot features across time.
            target: unused (kept for compatibility with the generic loss call
                    signature used by `compute_loss`).
            continuity_mask: [B, T, N] or [B, N] detached binary mask marking
                    slots that persist with the same identity. Values in {0, 1}
                    (accepts bool or float). MUST be detached from the graph.
        Returns:
            Scalar loss averaged across gaps by `gap_weights`.
        """
        del target  # unused

        slots = nn.functional.normalize(pred, p=2.0, dim=-1)
        # Robust to 3D [B, N, D] (validation path may collapse T axis).
        if slots.ndim == 3:
            slots = slots.unsqueeze(1)
        B, T, N, _ = slots.shape

        # Normalize continuity mask: ensure detached float tensor of shape [B, T, N].
        if continuity_mask is not None:
            continuity_mask = continuity_mask.detach()
            if continuity_mask.dtype == torch.bool:
                continuity_mask = continuity_mask.to(slots.dtype)
            else:
                continuity_mask = continuity_mask.to(slots.dtype)
            if continuity_mask.ndim == 2 and slots.ndim == 4:
                # Broadcast [B, N] across time.
                continuity_mask = continuity_mask.unsqueeze(1).expand(-1, T, -1)

        # Optional "require all gaps valid" mode: an anchor-slot is active only
        # if every requested gap-pair around it is mutually continuous.
        if self.require_all_gaps_valid and continuity_mask is not None:
            # anchor_valid[b, t, n] == 1 iff continuity_mask is 1 at t and at t+g for all g
            max_gap = max(self.gaps)
            if T <= max_gap:
                return slots.sum() * 0.0  # no valid anchor frames; return zero with grad.
            anchor_valid = continuity_mask[:, : T - max_gap].clone()
            for g in self.gaps:
                anchor_valid = anchor_valid * continuity_mask[:, g : g + (T - max_gap)]
        else:
            anchor_valid = None

        # Optionally merge batch into the slot dimension for batch-level negatives.
        if self.batch_contrast:
            slots_bc = slots.split(1)  # list of [1, T, N, D]
            slots_bc = torch.cat(slots_bc, dim=-2)  # [1, T, N*B, D]
            if continuity_mask is not None:
                mask_bc = continuity_mask.split(1)  # list of [1, T, N]
                mask_bc = torch.cat(mask_bc, dim=-1)  # [1, T, N*B]
            else:
                mask_bc = None
            if anchor_valid is not None:
                av_bc = anchor_valid.split(1)
                av_bc = torch.cat(av_bc, dim=-1)  # [1, T-max_gap, N*B]
            else:
                av_bc = None
        else:
            slots_bc = slots
            mask_bc = continuity_mask
            av_bc = anchor_valid

        Bc, Tc, Sc, _ = slots_bc.shape

        loss_total = slots_bc.sum() * 0.0  # zero tensor preserving grad/device/dtype
        total_weight = 0.0

        for g, w in zip(self.gaps, self.gap_weights):
            if Tc <= g:
                # Not enough frames for this gap; skip without penalty.
                continue

            s1 = slots_bc[:, : Tc - g, :, :]  # [Bc, Tc-g, Sc, D]
            s2 = slots_bc[:, g:, :, :]        # [Bc, Tc-g, Sc, D]
            # Similarity: anchor (t, n) against every slot (t+g, j) in the same frame.
            ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature  # [Bc, Tc-g, Sc, Sc]
            Bg, Tg, Sg, _ = ss.shape
            ss = ss.reshape(Bg * Tg, Sg, Sg)
            eye_target = torch.eye(Sg, device=ss.device, dtype=ss.dtype).expand(Bg * Tg, Sg, Sg)
            loss_per_slot = self.criterion(ss, eye_target)  # [Bg*Tg, Sg]

            # Build per-anchor validity mask.
            if mask_bc is not None:
                m1 = mask_bc[:, : Tc - g, :]  # [Bc, Tc-g, Sc]
                m2 = mask_bc[:, g:, :]        # [Bc, Tc-g, Sc]
                valid = (m1 * m2).reshape(Bg * Tg, Sg)
            else:
                valid = torch.ones(Bg * Tg, Sg, device=ss.device, dtype=ss.dtype)

            if av_bc is not None:
                # Crop anchor_valid to this gap's anchor slice.
                av_slice = av_bc[:, : Tc - g, :].reshape(Bg * Tg, Sg)
                valid = valid * av_slice

            valid_count = valid.sum()
            if valid_count > 0:
                gap_loss = (loss_per_slot * valid).sum() / valid_count
            else:
                # No valid anchor pair at this gap: contribute zero (no grad), skip weight.
                continue

            loss_total = loss_total + w * gap_loss
            total_weight += w

        if total_weight > 0:
            loss_total = loss_total / total_weight

        return loss_total


class HardNegSlotContrast(Loss):
    """Hard-negative-mined variant of `Slot_Slot_Contrastive_Loss` (Idea #010).

    Mirrors the adjacent-frame slot-slot InfoNCE setup of
    `Slot_Slot_Contrastive_Loss`: for each anchor slot `i` at frame `t`, the
    positive is slot `i` at frame `t+1`, and negatives are the other slots in
    the same paired frame (optionally pooled across the batch).

    The only structural change is the InfoNCE denominator: instead of treating
    all negatives uniformly, each negative term `exp(sim_{i,j})` is reweighted
    by a softmax over negative similarities with temperature `hard_neg_T`
    (default `0.05`). This concentrates gradient on the hardest (most similar)
    negatives, which is the intended remedy for similar-looking objects on
    YT-VIS (gap G8 in the round-25 brainstorm).

    Optionally, `top_k_negatives` restricts the negative set to the K most
    similar negatives per anchor (K is clamped to the number of available
    negatives). The interface (`pred_key`, `target_key`, `temperature`,
    `batch_contrast`, `keep_input_dim`, ...) is kept compatible with the base
    so that this loss can be swapped in via the usual builder.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        hard_neg_T: float = 0.05,
        top_k_negatives: Optional[int] = None,
        batch_contrast: bool = True,
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.temperature = temperature
        if hard_neg_T <= 0:
            raise ValueError(f"`hard_neg_T` must be > 0, got {hard_neg_T}.")
        self.hard_neg_T = hard_neg_T
        if top_k_negatives is not None and top_k_negatives <= 0:
            raise ValueError(
                f"`top_k_negatives` must be None or > 0, got {top_k_negatives}."
            )
        self.top_k_negatives = top_k_negatives
        self.batch_contrast = batch_contrast
        # Route via the default existence-mask dispatch path in
        # `SlotContrastVideoModel.compute_loss`.
        self.mask_key = mask_key

    def forward(self, pred, target=None, existence_mask=None, **kwargs):
        """
        Args:
            pred: [B, T, N, D] slot representations across time.
            target: unused (kept for compatibility with the generic
                `compute_loss` call signature).
            existence_mask: [B, T, N] or [B, N] with 1.0 for valid slots and
                0.0 for empty slots. Optional.
        Returns:
            Scalar InfoNCE-style loss with hard-negative reweighting.
        """
        del target  # unused

        slots = nn.functional.normalize(pred, p=2.0, dim=-1)

        # Handle 3D [B, N, D] by inserting T=1; matches base loss behavior on
        # validation paths where chunking collapses the time axis.
        if slots.ndim == 3:
            slots = slots.unsqueeze(1)  # [B, 1, N, D]

        # Align mask shape with slots: broadcast [B, N] across time if needed.
        if existence_mask is not None:
            if existence_mask.dtype == torch.bool:
                existence_mask = existence_mask.to(slots.dtype)
            else:
                existence_mask = existence_mask.to(slots.dtype)
            if existence_mask.ndim == 2 and slots.ndim == 4:
                existence_mask = existence_mask.unsqueeze(1).expand(-1, slots.shape[1], -1)

        # Optional batch pooling: merge per-video slots into one big slot set
        # so negatives include slots from other videos in the batch.
        if self.batch_contrast:
            slots_list = slots.split(1)  # list of [1, T, N, D]
            slots = torch.cat(slots_list, dim=-2)  # [1, T, N*B, D]
            if existence_mask is not None:
                mask_list = existence_mask.split(1)  # list of [1, T, N]
                existence_mask = torch.cat(mask_list, dim=-1)  # [1, T, N*B]

        # Adjacent-frame anchor/positive pairs.
        s1 = slots[:, :-1, :, :]  # [B, T-1, S, D]
        s2 = slots[:, 1:, :, :]   # [B, T-1, S, D]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature  # [B, T-1, S, S]
        Bg, Tg, Sg, _ = ss.shape
        ss = ss.reshape(Bg * Tg, Sg, Sg)  # [BT, S, S]

        if Sg <= 1:
            # No negatives available; return zero with grad attached.
            return slots.sum() * 0.0

        # Split diagonal (positive logits) from off-diagonal (negative logits).
        eye = torch.eye(Sg, device=ss.device, dtype=torch.bool)
        pos_logits = ss[:, eye]  # [BT, S] — diagonal entries
        # Negatives: same row, off-diagonal columns. Use masked_select-style
        # gather: drop the diagonal from each row.
        off_diag = (~eye).unsqueeze(0).expand(Bg * Tg, Sg, Sg)  # [BT, S, S]
        neg_logits = ss[off_diag].reshape(Bg * Tg, Sg, Sg - 1)  # [BT, S, S-1]

        # Optionally keep only the top-K most similar negatives per anchor.
        if self.top_k_negatives is not None:
            k = min(self.top_k_negatives, neg_logits.shape[-1])
            neg_logits, _ = torch.topk(neg_logits, k=k, dim=-1, largest=True, sorted=False)

        # Hard-negative weights: softmax over negatives with temperature
        # `hard_neg_T`. Using `ss` (already scaled by `temperature`) as the
        # similarity score; the inner division scales it by `hard_neg_T` so
        # that a smaller `hard_neg_T` sharpens toward the hardest negatives.
        # Detach weights to avoid double-counting gradient paths through the
        # attention weighting (standard practice for hard-negative mining).
        neg_weights = torch.softmax(neg_logits.detach() / self.hard_neg_T, dim=-1)  # [BT, S, K]

        # Reweighted negative aggregate (in log-space for numerical stability).
        # log( sum_j w_ij * exp(neg_logits_ij) ) = logsumexp(neg_logits + log w_ij, dim=-1)
        log_w = torch.log(neg_weights.clamp_min(1e-12))
        neg_lse = torch.logsumexp(neg_logits + log_w, dim=-1)  # [BT, S]

        # InfoNCE-style loss per anchor:
        # loss_i = -log( exp(pos) / ( exp(pos) + sum_j w_ij * exp(neg_ij) ) )
        #        = softplus(neg_lse - pos)
        loss_per_slot = torch.nn.functional.softplus(neg_lse - pos_logits)  # [BT, S]

        # Existence masking: anchor slot must exist at both t and t+1.
        if existence_mask is not None:
            m1 = existence_mask[:, :-1, :]  # [Bc, Tc-1, Sc]
            m2 = existence_mask[:, 1:, :]   # [Bc, Tc-1, Sc]
            valid = (m1 * m2).reshape(Bg * Tg, Sg)
            valid_count = valid.sum()
            if valid_count > 0:
                loss = (loss_per_slot * valid).sum() / valid_count
            else:
                loss = loss_per_slot.mean()
        else:
            loss = loss_per_slot.mean()

        return loss


class FBCycleAssignLoss(Loss):
    """Forward-Backward Cycle-Consistency Loss on slot assignments (Idea #009).

    For every adjacent pair of frames ``(t-1, t)`` we build two soft
    assignment matrices using temperature-scaled cosine similarity:

        P_fwd[b, t, i, j] = softmax_j( cos(slot_{t-1, i}, slot_{t, j}) / tau )
        P_bwd[b, t, i, j] = softmax_j( cos(slot_{t,   i}, slot_{t-1, j}) / tau )

    A consistent correspondence must survive the cycle ``t-1 -> t -> t-1``, so
    the composite assignment ``C = P_fwd @ P_bwd`` should approach the
    identity. We penalize the squared Frobenius deviation

        L = mean_{b, t} || C_{b, t} - I_N ||_F^2

    masking anchor rows of the composite by the joint
    ``existence_mask[t-1] * existence_mask[t]`` so that only slots that exist
    at both endpoints contribute to the loss.

    Notes:
        - Expects ``pred`` of shape ``[B, T, N, D]``; no patch / spatial dims.
        - ``target`` is unused and kept only for compatibility with the
          ``compute_loss`` dispatch signature shared by the other slot losses.
        - ``existence_mask`` may be provided as ``[B, T, N]`` or ``[B, N]``; a
          2-D mask is broadcast across time.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str = "dummy",
        temperature: float = 0.1,
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        if temperature <= 0:
            raise ValueError(f"`temperature` must be > 0, got {temperature}.")
        self.temperature = temperature
        self.mask_key = mask_key

    def forward(self, pred, target=None, existence_mask=None, **kwargs):
        """Compute the forward-backward cycle-consistency loss on assignments.

        Args:
            pred: ``[B, T, N, D]`` slot features across time.
            target: unused; kept for dispatch-signature compatibility.
            existence_mask: ``[B, T, N]`` or ``[B, N]`` float/bool tensor
                marking valid slots. If None, all slots are treated as valid.

        Returns:
            Scalar loss averaged over batch and adjacent-frame pairs.
        """
        del target  # unused

        slots = nn.functional.normalize(pred, p=2.0, dim=-1)

        # Handle 3D [B, N, D] robustly (validation path may collapse T axis).
        if slots.ndim == 3:
            slots = slots.unsqueeze(1)  # [B, 1, N, D]

        B, T, N, _ = slots.shape

        if T < 2:
            # No adjacent pair available; preserve grad/device/dtype.
            return slots.sum() * 0.0

        # Normalize existence mask to [B, T, N] float tensor on the slot device.
        if existence_mask is not None:
            existence_mask = existence_mask.detach()
            existence_mask = existence_mask.to(slots.dtype)
            if existence_mask.ndim == 2 and slots.ndim == 4:
                existence_mask = existence_mask.unsqueeze(1).expand(-1, T, -1)

        s_prev = slots[:, :-1, :, :]  # [B, T-1, N, D]
        s_next = slots[:, 1:, :, :]   # [B, T-1, N, D]

        # Slots are L2-normalized, so inner products = cosine similarities.
        sim_fwd = torch.matmul(s_prev, s_next.transpose(-2, -1)) / self.temperature
        sim_bwd = torch.matmul(s_next, s_prev.transpose(-2, -1)) / self.temperature

        P_fwd = torch.softmax(sim_fwd, dim=-1)  # [B, T-1, N, N]
        P_bwd = torch.softmax(sim_bwd, dim=-1)  # [B, T-1, N, N]

        cycle = torch.matmul(P_fwd, P_bwd)  # [B, T-1, N, N]

        eye = torch.eye(N, device=cycle.device, dtype=cycle.dtype)
        eye = eye.view(1, 1, N, N).expand_as(cycle)

        # Per-(b, t, i) squared row error; row i is the anchor slot at t-1
        # that traverses the cycle back to itself.
        row_sq_err = (cycle - eye).pow(2).sum(dim=-1)  # [B, T-1, N]

        if existence_mask is not None:
            m_prev = existence_mask[:, :-1, :]  # [B, T-1, N]
            m_next = existence_mask[:, 1:, :]   # [B, T-1, N]
            valid = m_prev * m_next             # [B, T-1, N]

            valid_count = valid.sum()
            if valid_count > 0:
                loss = (row_sq_err * valid).sum() / valid_count
            else:
                # No valid anchors: return zero with grad/device preserved.
                loss = row_sq_err.mean() * 0.0
        else:
            loss = row_sq_err.mean()

        return loss


class DynamicsLoss(Loss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rollout_length = prediction.shape[1]
        target = target[:, -rollout_length:]
        loss = self.criterion(prediction, target)
        return loss


class EntropyLoss(Loss):
    def __init__(self, pred_key: str, target_key: str = "dummy", **kwargs):
        kwargs.pop('video_inputs', None)
        kwargs.pop('patch_inputs', None)
        kwargs.pop('keep_input_dim', None)
        super().__init__(pred_key, target_key, video_inputs=False, patch_inputs=False, keep_input_dim=True, **kwargs)
    
    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)
    
    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return None
    
    def forward(self, entropy_loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return entropy_loss


class CycleConsistencyLoss(Loss):
    """Cycle/Temporal Cross-Consistency Loss for re-slotting.
    
    When window=0: Same-frame cycle consistency
    When window>0: Temporal cross-consistency with random sampling
    
    Computes MSE between cycle slots and target slots (detached).
    """
    def __init__(
        self,
        pred_key: str = "processor.cycle_slots",
        target_key: str = "processor.cycle_targets",
        **kwargs,
    ):
        kwargs.pop('video_inputs', None)
        kwargs.pop('patch_inputs', None)
        kwargs.pop('keep_input_dim', None)
        super().__init__(pred_key, target_key, video_inputs=False, patch_inputs=False, keep_input_dim=True, **kwargs)
        self.criterion = nn.MSELoss()
    
    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)
    
    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.target_path)
    
    def forward(self, cycle_slots: torch.Tensor, target_slots: torch.Tensor) -> torch.Tensor:
        return self.criterion(cycle_slots, target_slots)


class SlotDropRecoverLoss(Loss):
    """Slot-Drop-Recover loss (Idea #012).

    Training-time occlusion simulation. At each step a random subset of the
    previous-frame slots is dropped (zeroed) before being passed to the
    corrector; the corrector is expected to recover the dropped identities
    from current-frame evidence. This loss penalizes recovered slots whose
    features diverge from the (detached) pre-drop slot features at matching
    positions.

    Let ``pred`` denote the current-frame slots ``[B, T, N, D]`` produced by
    the corrector after receiving the dropped previous slots, and let
    ``pre_drop_slots`` denote the detached copy of the same tensor captured
    before the drop. For every ``(b, t, n)`` indicated by ``drop_mask`` we
    compute

        l_{b, t, n} = 1 - cos(pred[b, t, n], pre_drop_slots[b, t, n].detach())

    and average over the dropped positions. When ``drop_mask`` is not
    provided (or selects no positions), the loss is zero with grad/device/dtype
    preserved via ``pred.sum() * 0.0``.

    Notes:
        - ``pred`` and ``pre_drop_slots`` are expected to share shape
          ``[B, T, N, D]``.
        - ``drop_mask`` has shape ``[B, T, N]`` and is interpreted as a 0/1
          indicator of positions where a pre-drop slot was zeroed out. It is
          passed via ``kwargs`` by the caller in the processor forward.
        - ``existence_mask`` (if provided) is combined with ``drop_mask`` so
          that only dropped, *existing* slot positions contribute.
        - ``target`` is unused and retained for signature compatibility with
          the shared ``compute_loss`` dispatch.
    """

    # Declare auxiliary tensors this loss consumes from the model outputs dict.
    # Consumed by `models.ObjectCentricModel.compute_loss` (round-26 B1) which
    # looks up `outputs[key]` for each name here and forwards them as kwargs.
    aux_keys = ["pre_drop_slots", "drop_mask"]

    def __init__(
        self,
        pred_key: str,
        target_key: str = "dummy",
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        # Force the shape-preserving path: this loss operates directly on
        # ``[B, T, N, D]`` slot tensors and must not be flattened by the base
        # ``to_canonical_dims`` reshape.
        kwargs.pop('video_inputs', None)
        kwargs.pop('patch_inputs', None)
        kwargs.pop('keep_input_dim', None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        self.mask_key = mask_key

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        # Bypass the base class dim-check/canonicalization; we need the raw
        # ``[B, T, N, D]`` slot tensor.
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        # ``target`` is unused by this loss (see forward signature). Return
        # ``None`` rather than attempting to resolve the placeholder key.
        return None

    def forward(
        self,
        pred,
        target=None,
        existence_mask=None,
        pre_drop_slots=None,
        **kwargs,
    ):
        """Compute the slot-drop-recover loss.

        Args:
            pred: ``[B, T, N, D]`` current-frame slot features produced by the
                corrector after the drop.
            target: unused; kept for dispatch-signature compatibility.
            existence_mask: optional ``[B, T, N]`` or ``[B, N]`` float/bool
                tensor marking valid slots. If provided, restricts the loss to
                positions that both were dropped and exist.
            pre_drop_slots: ``[B, T, N, D]`` detached snapshot of the slots
                *before* the drop. Used as the recovery target. Detached again
                here for safety.
            drop_mask: ``[B, T, N]`` (passed via ``kwargs``) 0/1 indicator of
                positions that were dropped. If missing or selects no
                positions, the loss is zero.

        Returns:
            Scalar loss averaged over dropped (and existing) positions.
        """
        del target  # unused

        drop_mask = kwargs.get("drop_mask", None)

        # If no drop happened this step, return zero with grad/device preserved.
        if drop_mask is None or pre_drop_slots is None:
            return pred.sum() * 0.0

        if pred.shape != pre_drop_slots.shape:
            raise ValueError(
                "SlotDropRecoverLoss expects `pred` and `pre_drop_slots` to "
                f"share shape, got {tuple(pred.shape)} vs "
                f"{tuple(pre_drop_slots.shape)}."
            )

        # Detach the target to ensure no gradient flows back through the
        # pre-drop snapshot, per the contract documented in the class docstring.
        target_slots = pre_drop_slots.detach()

        B, T, N, _ = pred.shape

        # Normalize drop_mask to float on the pred device.
        drop_mask = drop_mask.to(device=pred.device, dtype=pred.dtype)
        if drop_mask.shape != (B, T, N):
            raise ValueError(
                "SlotDropRecoverLoss expects `drop_mask` of shape "
                f"[B, T, N]={B, T, N}, got {tuple(drop_mask.shape)}."
            )

        # Combine with existence mask when available so that only valid,
        # dropped slot positions contribute to the loss.
        if existence_mask is not None:
            emask = existence_mask.detach().to(device=pred.device, dtype=pred.dtype)
            if emask.ndim == 2 and pred.ndim == 4:
                emask = emask.unsqueeze(1).expand(-1, T, -1)
            drop_mask = drop_mask * emask

        # Cosine similarity per (b, t, n) on the feature dim.
        pred_n = nn.functional.normalize(pred, p=2.0, dim=-1)
        tgt_n = nn.functional.normalize(target_slots, p=2.0, dim=-1)
        cos_sim = (pred_n * tgt_n).sum(dim=-1)  # [B, T, N]
        elem_loss = 1.0 - cos_sim               # [B, T, N]

        valid_count = drop_mask.sum()
        if valid_count > 0:
            loss = (elem_loss * drop_mask).sum() / valid_count
        else:
            # No dropped positions: return zero with grad/device preserved.
            loss = elem_loss.mean() * 0.0

        return loss


class MaskIoUQualityLoss(Loss):
    """Per-slot mask-quality prediction loss (Idea #023).

    Trains the ``MaskIoUQualityHead`` to predict each slot's mask IoU against a
    target quality signal. Two configurations are supported upstream:

      - Table B: ``target`` is the IoU between each slot's decoder mask and the
        matched ground-truth mask.
      - Table A: ``target`` is the slot's decoder self-agreement IoU, computed
        upstream without any GT (e.g. teacher/student agreement). The caller is
        responsible for producing this tensor; THIS loss class does NOT compute
        self-agreement itself.

    In both cases the target is a scalar in ``[0, 1]`` per slot per frame. The
    prediction tensor is the head's sigmoid output (``quality`` in
    ``[0, 1]``). We recover the underlying logits via ``logit = sigmoid^{-1}``
    and apply binary cross-entropy with logits against the target, masked by
    the ``existence_mask``.

    Shapes:
        - ``pred``: ``[B, T, N]`` in ``[0, 1]`` (quality from the head).
        - ``target``: ``[B, T, N]`` in ``[0, 1]`` (mask_iou_target).
        - ``existence_mask``: ``[B, T, N]`` or ``[B, N]`` float/bool.

    Returns:
        Scalar mean BCE averaged over existing slot positions. If no valid
        positions are available, returns zero with grad/device preserved.
    """

    def __init__(
        self,
        pred_key: str = "quality",
        target_key: str = "mask_iou",
        mask_key: str = "existence_mask",
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        # Retain the literal key strings explicitly; the base class stores them
        # split as `pred_path` / `target_path`, but the task contract wires the
        # loss infra through `pred_key` / `target_key` attributes.
        self.pred_key = pred_key
        self.target_key = target_key
        self.mask_key = mask_key
        if not (0.0 < eps < 1.0):
            raise ValueError(f"`eps` must be in (0, 1), got {eps}.")
        self.eps = float(eps)

    def forward(self, pred, target=None, existence_mask=None, **kwargs):
        """Compute the mask-IoU quality BCE loss.

        Args:
            pred: ``[B, T, N]`` per-slot quality in ``[0, 1]`` (sigmoid output
                from ``MaskIoUQualityHead``).
            target: ``[B, T, N]`` per-slot target IoU in ``[0, 1]``. Must be
                provided (either GT mask IoU or pre-computed decoder
                self-agreement IoU). If ``None`` the loss returns zero.
            existence_mask: ``[B, T, N]`` or ``[B, N]`` float/bool marking
                valid slots. If provided, the loss averages only over valid
                positions.

        Returns:
            Scalar BCE loss. Zero (with grad/device preserved) when no valid
            target or no valid positions are available.
        """
        del kwargs  # any additional kwargs are ignored.

        # No target: preserve grad/device/dtype and return zero.
        if target is None:
            return pred.sum() * 0.0

        target = target.detach().to(device=pred.device, dtype=pred.dtype)
        if target.shape != pred.shape:
            raise ValueError(
                "MaskIoUQualityLoss expects `pred` and `target` to share "
                f"shape, got {tuple(pred.shape)} vs {tuple(target.shape)}."
            )

        # Recover logits from the sigmoid output: sigmoid^{-1}(p) = log(p / (1 - p)).
        p = pred.clamp(min=self.eps, max=1.0 - self.eps)
        logits = torch.log(p) - torch.log1p(-p)

        # Clamp the target into [eps, 1 - eps] as well; BCE accepts soft labels
        # in [0, 1] but we guard against numerical edge cases.
        t = target.clamp(min=0.0, max=1.0)

        # Per-element BCE with logits (soft labels allowed).
        elem_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, t, reduction="none"
        )  # [B, T, N]

        if existence_mask is not None:
            emask = existence_mask.detach().to(device=pred.device, dtype=pred.dtype)
            if emask.ndim == 2 and pred.ndim == 3:
                emask = emask.unsqueeze(1).expand_as(pred)
            if emask.shape != pred.shape:
                raise ValueError(
                    "MaskIoUQualityLoss expects `existence_mask` broadcastable "
                    f"to {tuple(pred.shape)}, got {tuple(emask.shape)}."
                )
            valid_count = emask.sum()
            if valid_count > 0:
                loss = (elem_loss * emask).sum() / valid_count
            else:
                # No valid positions: return zero with grad/device preserved.
                loss = elem_loss.mean() * 0.0
        else:
            loss = elem_loss.mean()

        return loss


class FlowConsistencyMaskLoss(Loss):
    """Flow-Consistency Mask Regularizer (Idea T3-02, FCMR).

    Penalizes decoder masks that do not transport cleanly under the
    pre-computed forward optical flow. For every adjacent pair of frames
    ``(t, t+1)`` we warp ``Mask_t`` by the forward flow ``flow_{t -> t+1}``
    into frame ``t+1``'s coordinate system, compare against the Hungarian-
    identity-aligned mask ``Mask_{t+1}``, and sum the squared error over
    patches / slots / time. No architectural change, no inference-time
    dependency: flow is detached and used as a training-side consistency
    signal only (opposite of v9's flow-conditioned predictor branch).

    Concretely::

        L = (1 / (T-1)) * sum_t  MSE(warp(M_t, flow_t) * visibility_{t+1},
                                     M_{t+1}^{matched})

    where ``warp`` uses ``F.grid_sample`` with ``align_corners=False``,
    ``mode='bilinear'``, ``padding_mode='zeros'`` and ``visibility_{t+1}``
    is the set of destination pixels whose sampling grid falls inside the
    image bounds [-1, 1]^2 (dropping samples that left the frame, which
    carry no signal).

    Shapes:
        - ``pred``: ``[B, T, K, P]`` (patch-flattened decoder masks from the
          :class:`SlotMixerDecoder` / :class:`MLPDecoder` path) or
          ``[B, T, K, H, W]`` (spatial decoder path). Patches are reshaped
          to ``[H, W]`` with ``H = W = sqrt(P)``.
        - ``forward_flow``: ``[B, T-1, H_f, W_f, 2]`` or ``[B, T, H_f, W_f, 2]``
          (padded variant; the last entry is unused). Values are dx/dy in
          pixel units at the flow's native resolution ``(H_f, W_f)``; the
          loss resamples flow to the mask's ``(H, W)`` and converts to the
          normalized ``[-1, 1]`` grid expected by ``grid_sample``.
        - ``hungarian_match_indices``: optional ``List[Optional[Tensor[B, K]]]``
          of length ``T`` whose ``t``-th entry is the per-batch permutation
          reordering frame-``(t-1)``'s slots into frame-``t``'s slot order
          (convention from
          :meth:`slotcontrast.modules.networks.HungarianPredictor._hungarian_match`:
          ``indices[b, i] = col_ind`` such that ``reordered[b, i] = curr[b, col_ind[i]]``).
          The first entry is ``None``. When omitted we assume the upstream
          predictor has already reordered slots in-place (the common path
          when ``predictor.name`` is Hungarian-family: ``_prev_slots`` is set
          to the reordered tensor, so ``pred[:, t, k]`` and ``pred[:, t+1, k]``
          already share identity ``k``). When both conditions hold the loss
          is a no-op on identity.

    Flow is *required* — this loss follows the no-fallback contract: if
    ``forward_flow`` is missing the model forward is expected to have kept it
    out of ``outputs`` and the dispatcher will not populate the kwarg; in
    that case :meth:`forward` raises :class:`ValueError` so the problem is
    loud rather than silent.
    """

    # Consumed by `models.ObjectCentricModel.compute_loss` via the `aux_keys`
    # dispatch: for each name here the dispatcher looks up `outputs[key]` and
    # forwards it as a kwarg. Both keys are lifted from the top-level outputs
    # dict in `models.forward` (mirroring the `pre_drop_slots` / `drop_mask`
    # lift pattern around `models.py:351-354`).
    aux_keys = ["forward_flow", "hungarian_match_indices"]

    def __init__(
        self,
        pred_key: str = "decoder.masks",
        target_key: str = "dummy",
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        # Force the shape-preserving path; we operate directly on the raw
        # ``[B, T, K, P]`` (or ``[B, T, K, H, W]``) decoder mask tensor and
        # must NOT be flattened by the base ``to_canonical_dims`` reshape
        # (which would collapse the time axis).
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        self.mask_key = mask_key

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        # Bypass the base class dim-check / canonicalization; we need the
        # raw mask tensor with all spatial / slot axes intact.
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        # Target is synthesized inside ``forward`` from the same mask tensor
        # warped by flow; no separate target tensor is needed.
        return None

    @staticmethod
    def _build_sampling_grid(H: int, W: int, flow_hw: torch.Tensor) -> torch.Tensor:
        """Convert pixel-unit forward flow at (H, W) to a grid_sample grid.

        Semantics: ``forward_flow[b, y, x] = (dx, dy)`` means the source
        pixel ``(x, y)`` in frame ``t`` moves to ``(x + dx, y + dy)`` in
        frame ``t+1``. We want to compute ``M_{t+1}_warp_from_t`` — a
        backward-sampling warp of ``M_t`` into frame ``t+1``'s coordinate
        system. Using the standard "backward warp with forward flow"
        approximation (exact under rigid translation, first-order-accurate
        otherwise), the source location for destination pixel ``(x_d, y_d)``
        in frame ``t+1`` is ``(x_d - dx(x_d, y_d), y_d - dy(x_d, y_d))``.

        Args:
            H, W: target spatial resolution (same as mask resolution).
            flow_hw: ``[B, H, W, 2]`` forward flow in pixel units at
                resolution ``(H, W)`` with last dim storing ``(dx, dy)``.

        Returns:
            ``[B, H, W, 2]`` grid in the normalized ``[-1, 1]`` frame expected
            by ``F.grid_sample(align_corners=False)`` with PyTorch's
            ``(x, y)`` ordering on the last axis.
        """
        B = flow_hw.shape[0]
        device = flow_hw.device
        dtype = flow_hw.dtype

        # Destination-pixel centers in normalized coords under
        # align_corners=False: centers sit at (2*j + 1)/W - 1 for j in 0..W-1.
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (2.0 / H) - 1.0
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (2.0 / W) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W] each
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2] in (x, y)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Pixel-unit dx/dy -> normalized delta via division by (W/2, H/2).
        # align_corners=False matches the half-pixel offsets encoded in
        # base_grid above. Source = destination - flow (see docstring).
        scale = torch.tensor([2.0 / W, 2.0 / H], device=device, dtype=dtype)
        flow_norm = flow_hw * scale  # [B, H, W, 2]

        return base_grid - flow_norm

    def forward(
        self,
        pred,
        target=None,
        existence_mask=None,
        forward_flow=None,
        hungarian_match_indices: Optional[List[Optional[torch.Tensor]]] = None,
        **kwargs,
    ):
        """Compute the flow-consistency mask regularizer.

        Args:
            pred: decoder masks; either ``[B, T, K, P]`` (patch-flattened) or
                ``[B, T, K, H, W]`` (spatial). No gradient is stopped here —
                the loss *is* the mask-quality training signal.
            target: unused; preserved for dispatch-signature parity.
            existence_mask: unused in the core computation (masks from dead
                slots are already zeroed upstream by ``SlotMixerDecoder``'s
                ``masked_fill(-inf)`` before softmax); accepted for
                dispatch-signature parity.
            forward_flow: required; ``[B, T-1, H_f, W_f, 2]`` or
                ``[B, T, H_f, W_f, 2]`` (padded; last entry unused).
            hungarian_match_indices: optional length-``T`` list whose ``t``-th
                entry is ``[B, K]`` or ``None``. Used to permute
                ``pred[:, t]`` into ``pred[:, t+1]``'s slot order. When the
                upstream Hungarian path already baked reorder into the slot
                tensor, the dispatcher typically passes ``None`` for every
                entry (which is the identity); this branch then short-circuits.

        Returns:
            Scalar mean MSE averaged over (B, T-1, K, H, W) positions.

        Raises:
            ValueError: if ``forward_flow`` is ``None`` (no silent fallback).
        """
        del target, existence_mask, kwargs

        if forward_flow is None:
            # No-fallback: surface the missing-flow condition loudly. The
            # dispatcher in `models.compute_loss` only forwards aux kwargs
            # when the corresponding `outputs[key]` is set; a missing
            # `forward_flow` here means the batch never carried flow (e.g.
            # MOVi-D without Kubric GT flow, or YT-VIS without RAFT cache)
            # and the user should either wire the loader or remove this loss.
            raise ValueError(
                "FlowConsistencyMaskLoss requires `forward_flow` in the batch "
                "(lifted to outputs['forward_flow'] by the model forward). "
                "No fallback is provided: either precompute RAFT / Kubric "
                "forward flow, or disable this loss in the config."
            )

        # Normalize mask shape to [B, T, K, H, W].
        if pred.ndim == 5:
            B, T, K, H, W = pred.shape
            masks = pred
        elif pred.ndim == 4:
            B, T, K, P = pred.shape
            side = int(round(math.sqrt(P)))
            if side * side != P:
                raise ValueError(
                    f"FlowConsistencyMaskLoss expects square patch grids; "
                    f"got P={P} which is not a perfect square."
                )
            H = W = side
            masks = pred.view(B, T, K, H, W)
        else:
            raise ValueError(
                "FlowConsistencyMaskLoss expects `pred` of shape "
                "[B, T, K, P] or [B, T, K, H, W]; "
                f"got ndim={pred.ndim} shape={tuple(pred.shape)}."
            )

        if T < 2:
            # No adjacent pair: return zero with grad/device/dtype preserved.
            return masks.sum() * 0.0

        # Flow is a frozen training signal — detach once up front.
        flow = forward_flow.detach()
        if flow.ndim != 5 or flow.shape[-1] != 2:
            raise ValueError(
                "FlowConsistencyMaskLoss expects `forward_flow` of shape "
                "[B, T_flow, H_f, W_f, 2]; "
                f"got shape {tuple(flow.shape)}."
            )
        if flow.shape[0] != B:
            raise ValueError(
                "FlowConsistencyMaskLoss: flow batch size "
                f"{flow.shape[0]} != mask batch size {B}."
            )
        flow_len = flow.shape[1]
        if flow_len < T - 1:
            raise ValueError(
                "FlowConsistencyMaskLoss: `forward_flow` time dim "
                f"{flow_len} < T-1 = {T - 1}; expected per-step forward flow."
            )

        # Resample flow to the mask's spatial resolution (H, W). Flow pixel
        # units scale with resolution, so the pixel-unit values must be
        # rescaled by (H/H_f, W/W_f) after bilinear interpolation.
        H_f, W_f = flow.shape[2], flow.shape[3]
        if (H_f, W_f) != (H, W):
            # [B, T_flow, H_f, W_f, 2] -> [(B*T_flow), 2, H_f, W_f] for interp.
            flow_b = flow.permute(0, 1, 4, 2, 3).reshape(B * flow_len, 2, H_f, W_f)
            flow_b = nn.functional.interpolate(
                flow_b.float(), size=(H, W), mode="bilinear", align_corners=False
            ).to(flow.dtype)
            flow_b = flow_b.view(B, flow_len, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
            # Pixel-unit rescale to new resolution.
            sx = float(W) / float(W_f)
            sy = float(H) / float(H_f)
            flow_b[..., 0] = flow_b[..., 0] * sx
            flow_b[..., 1] = flow_b[..., 1] * sy
            flow = flow_b
        # After this point flow is [B, flow_len, H, W, 2] in pixel units at (H, W).

        # Hungarian reorder has already been applied upstream: HungarianPredictor
        # (and the MAP-with-reject family) store reordered slots as
        # ``_prev_slots`` and emit them as ``processor.state``, so the decoder
        # sees masks that are ALREADY in identity order — i.e. ``masks[:, t, k]``
        # and ``masks[:, t+1, k]`` share identity ``k`` across adjacent frames.
        # Re-applying ``hungarian_match_indices[t+1]`` here would permute a
        # SECOND time and supervise slot ``i`` against the wrong target for any
        # non-identity match (the original Codex-review CRITICAL #1). We
        # therefore ignore ``hungarian_match_indices`` entirely: it is kept in
        # the signature only because ``aux_keys`` drives the dispatcher and
        # downstream models.py relies on that contract.
        del hungarian_match_indices

        # Build the warp sampling grid once per (B, t) pair. Accumulators are
        # forced to fp32 (MAJOR #7) to avoid fp16 overflow / loss-of-precision
        # under ``torch.autocast``; ``masks`` stays in its native dtype so the
        # backward graph is unchanged.
        loss_sum = torch.zeros((), dtype=torch.float32, device=masks.device)
        total_count = torch.zeros((), dtype=torch.float32, device=masks.device)

        for t in range(T - 1):
            flow_t = flow[:, t]  # [B, H, W, 2]
            grid_tp1 = self._build_sampling_grid(H, W, flow_t)  # [B, H, W, 2]

            # Visibility mask: destination samples whose source coordinate
            # falls inside the normalized image bounds [-1, 1]. Out-of-bound
            # samples are zero-padded by grid_sample and carry no signal; we
            # also zero the target so they do not contribute to the MSE.
            visibility = (
                (grid_tp1[..., 0] >= -1.0) & (grid_tp1[..., 0] <= 1.0) &
                (grid_tp1[..., 1] >= -1.0) & (grid_tp1[..., 1] <= 1.0)
            ).to(masks.dtype)  # [B, H, W]
            visibility = visibility.unsqueeze(1)  # [B, 1, H, W]

            # Warp M_t from frame-t to frame-(t+1) coordinates.
            m_t = masks[:, t]  # [B, K, H, W]
            warped_t = nn.functional.grid_sample(
                m_t, grid_tp1,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )  # [B, K, H, W]

            # Decoder masks are already in identity order across adjacent
            # frames (see note above on upstream Hungarian reorder); no slot
            # permutation is applied here.
            m_tp1 = masks[:, t + 1]  # [B, K, H, W]

            # Flow-consistency squared error, masked to in-frame destinations.
            diff_sq = (warped_t * visibility - m_tp1 * visibility).pow(2)
            loss_sum = loss_sum + diff_sq.float().sum()
            # MAJOR #4: denominator must count only visible positions (not
            # every slot/pixel), so that ``visibility`` zeroing the numerator
            # does not downweight sequences with more out-of-frame motion.
            # ``visibility`` is [B, 1, H, W]; broadcast to [B, K, H, W] by the
            # per-slot count K.
            valid_count = visibility.sum().to(torch.float32) * float(m_tp1.shape[1])
            total_count = total_count + valid_count

        # Sum-of-squares / visible-elements -> mean squared error averaged over
        # batch, time, slot, and in-frame spatial positions only.
        loss = loss_sum / total_count.clamp(min=1.0)
        # Preserve the prediction's dtype on the way out so the backward graph
        # keeps parity with the autocast region that produced ``masks``.
        return loss.to(masks.dtype)


class SaliencyTemporalConsistencyLoss(Loss):
    """Saliency-Temporal-Consistency Regularizer (Round-24 Fix #2B).

    Trains the *student* encoder so that per-patch saliency is temporally
    consistent under flow warping. Concretely, for every adjacent frame pair
    ``(t, t+1)`` we compute saliency ``S_t = ||feat_t||_2`` (or ``feat_t``
    cosine similarity against frame-mean), warp ``S_t`` into frame-``(t+1)``
    coordinates using the precomputed forward flow, and penalize the squared
    error against ``S_{t+1}``::

        L = (1 / (T-1)) * sum_t  MSE(warp(S_t, flow_t), S_{t+1}) * visibility

    Why this matters for the universal plug-in goal:
        Under per-frame greedy slot init (the GCv1 paper setup), slot
        identity at frame ``t+1`` depends on which patches the saliency map
        ranks highest. If saliency is temporally inconsistent, each frame's
        greedy pick lands on different patches, slot indices scramble, and
        Hungarian matching becomes essential. By training student features
        toward temporally-stable saliency, the per-frame greedy mechanism
        naturally lands on the same patches across frames. Slot identity
        persists from feature design, so Hungarian becomes redundant
        ("LoRA replaces matching") — the architectural prerequisite for
        Claim 3 (``GCv1 nomatch + LoRA > GCv1 matched``).

    The loss does NOT require existing Hungarian alignment of slots — it
    operates entirely on the per-patch feature stream from the encoder, so
    it can be trained alongside `skip_predictor=true` (no matching) configs.

    Shapes:
        - ``pred``: per-patch features ``[B, T, N, D]`` from the student
          encoder. The loss reads ``outputs["encoder"]["features"]`` by
          default (set ``pred_key`` to override).
        - ``forward_flow``: ``[B, T-1, H_f, W_f, 2]`` or padded
          ``[B, T, H_f, W_f, 2]`` (last entry unused).
    """

    aux_keys = ["forward_flow"]

    def __init__(
        self,
        pred_key: str = "encoder.features",
        target_key: str = "dummy",
        saliency_mode: str = "norm",
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        if saliency_mode not in ("norm",):
            raise ValueError(
                f"saliency_mode={saliency_mode!r} not implemented; only 'norm' supported"
            )
        self.saliency_mode = saliency_mode

    def get_prediction(self, outputs):
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs, outputs):
        return None

    def forward(self, pred, target=None, forward_flow=None, **kwargs):
        del target, kwargs
        if forward_flow is None:
            raise ValueError(
                "SaliencyTemporalConsistencyLoss requires `forward_flow` in the "
                "batch (lifted to outputs['forward_flow'] by model.forward)."
            )
        if pred.ndim != 4:
            raise ValueError(
                f"Expected per-patch video features [B, T, N, D]; got shape {tuple(pred.shape)}"
            )
        B, T, N, D = pred.shape
        side = int(round(math.sqrt(N)))
        if side * side != N:
            raise ValueError(
                f"Expected square patch grid (sqrt({N}) integer); got {N}"
            )

        # Saliency = per-patch feature L2 norm. Shape [B, T, side, side].
        saliency = pred.float().norm(dim=-1).reshape(B, T, side, side)

        # Resize forward_flow to (side, side) and convert from pixel units
        # at the original (Hf, Wf) to pixel units at (side, side).
        if forward_flow.ndim != 5 or forward_flow.shape[-1] != 2:
            raise ValueError(
                f"forward_flow must be [B, T_pair, Hf, Wf, 2]; got {tuple(forward_flow.shape)}"
            )
        # Slice off the padded trailing entry if caller used [B, T, Hf, Wf, 2].
        if forward_flow.shape[1] == T:
            flow = forward_flow[:, : T - 1]
        else:
            flow = forward_flow[:, : T - 1] if forward_flow.shape[1] > T - 1 else forward_flow
        Hf, Wf = flow.shape[2], flow.shape[3]

        # Resample to (side, side) via bilinear; flow is reshaped to
        # [B*(T-1), 2, Hf, Wf] for F.interpolate, then back.
        flow_btc = flow.reshape(B * (T - 1), Hf, Wf, 2).permute(0, 3, 1, 2).float()
        flow_btc = nn.functional.interpolate(
            flow_btc, size=(side, side), mode="bilinear", align_corners=False
        )
        flow_btc = flow_btc.permute(0, 2, 3, 1)  # [B*(T-1), side, side, 2]
        # Scale dx/dy from the native flow grid to the saliency-grid pixel
        # spacing: flow in pixel units must be rescaled by (side / Wf) for dx
        # and (side / Hf) for dy.
        scale = torch.tensor([side / Wf, side / Hf], device=flow_btc.device, dtype=flow_btc.dtype)
        flow_btc = flow_btc * scale
        flow_resized = flow_btc.reshape(B, T - 1, side, side, 2)

        loss_sum = saliency.new_zeros(())
        total_count = saliency.new_zeros(())
        for t in range(T - 1):
            grid_tp1 = FlowConsistencyMaskLoss._build_sampling_grid(side, side, flow_resized[:, t])

            visibility = (
                (grid_tp1[..., 0] >= -1.0) & (grid_tp1[..., 0] <= 1.0) &
                (grid_tp1[..., 1] >= -1.0) & (grid_tp1[..., 1] <= 1.0)
            ).to(saliency.dtype).unsqueeze(1)  # [B, 1, side, side]

            s_t = saliency[:, t].unsqueeze(1)  # [B, 1, side, side]
            s_tp1 = saliency[:, t + 1].unsqueeze(1)
            warped = nn.functional.grid_sample(
                s_t, grid_tp1,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )
            diff_sq = (warped * visibility - s_tp1 * visibility).pow(2)
            loss_sum = loss_sum + diff_sq.float().sum()
            total_count = total_count + visibility.sum().to(torch.float32)

        loss = loss_sum / total_count.clamp(min=1.0)
        return loss.to(pred.dtype)


class GSRSIdentityLoss(Loss):
    """Generative Slot Replay with Structured Identity Supervision (GSRS Part B).

    Implements proposal §3.10 (rectangular dustbin-assignment loss) + §3.11
    (replay-consistency loss) + §3.13 (unified inference head supervision).
    Given a per-batch replay sample produced by the frozen slot-compositional
    renderer ``g_φ`` (Part A) + a set of event labels defining the
    intervention, this loss supervises the main pipeline ``M_θ`` to (a)
    recover the replay-labelled slot trajectories in *identity* via a
    rectangular LSAP with born/dead dustbins, and (b) emit the correct
    categorical event at each recovered slot via the ``OpenSetIdentityHead``.

    Input contract (see ``models.py`` forward for how these are lifted into
    ``outputs``):

    * ``pred`` (``processor.state``): per-slot state tensor of shape
      ``[B, T, K_pred, D]`` produced by the student corrector/predictor.
    * ``outputs["open_set_probs"]``: the open-set head's softmax output of
      shape ``[B, T, K_pred, 4]`` over ``(source, dormant, null, born)``.
      Lifted by the model forward when ``self.open_set_head`` is configured.
    * ``replay_target_trajectories``: ``[B, T, K_target + K_dustbin, D]``
      target slot trajectories. The first ``K_target`` rows are *real*
      target slots produced by the renderer (modified teacher trajectories
      under the chosen intervention). The last ``K_dustbin`` rows are
      virtual dustbin rows; ``K_dustbin ∈ {2, 4}`` — one "born" row and one
      "dead" row at a minimum, optionally doubled to allow multi-dustbin
      assignment on long clips.
    * ``replay_event_flags``: ``[B, T, K_target + K_dustbin]`` integer
      labels in ``{0, 1, 2, 3}`` indexed by
      ``OpenSetIdentityHead.CATEGORIES`` — i.e. ``0 = source``,
      ``1 = dormant``, ``2 = null``, ``3 = born``. Dustbin rows use the
      same code (e.g. dead-dustbin row uses index 2 = null; born-dustbin
      row uses index 3 = born).
    * ``replay_family``: per-batch string label identifying which
      intervention family produced this sample (see §3.2:
      ``"same", "swap", "gap-return", "kill", "birth"``). Passed through
      as a plain Python list of length ``B`` on the model outputs dict.

    The loss raises ``ValueError`` on any missing aux key — no silent
    fallback — per the project-wide no-fallback directive.

    Output contract:

    * Scalar tensor ``L_total = L_assign + λ_event · L_event`` returned
      from ``forward``. The individual ``L_assign`` / ``L_event`` values
      are additionally stashed on ``self._last_components`` so upstream
      logging can pick them up without changing the dispatch signature
      (mirroring the pattern used by other aux-bearing losses).
    """

    # Consumed by ``models.ObjectCentricModel.compute_loss`` via the
    # ``aux_keys`` dispatch: for each name here the dispatcher looks up
    # ``outputs[key]`` and forwards it as a kwarg. All three are lifted
    # into the top-level outputs dict by ``ObjectCentricModel.forward``
    # whenever the batch carries replay data (gated on ``replay_family``
    # being present in ``inputs``).
    aux_keys = ["replay_target_trajectories", "replay_event_flags", "replay_family", "open_set_probs"]

    # Event category → index mapping MUST match
    # ``OpenSetIdentityHead.CATEGORIES`` AND the sampler's ``EVENT_TO_INDEX``
    # in ``slotcontrast.modules.replay``. The authoritative source is
    # ``replay.EVENT_TO_INDEX`` (imported at module top as
    # ``_SHARED_EVENT_TO_INDEX``); this class-level copy is populated from
    # that import so the sampler, collator, and loss cannot silently drift
    # apart.
    EVENT_TO_INDEX: Dict[str, int] = dict(_SHARED_EVENT_TO_INDEX)

    def __init__(
        self,
        pred_key: str = "processor.state",
        target_key: str = "dummy",
        mask_key: str = "existence_mask",
        lambda_event: float = 0.2,
        c_bin: float = 0.5,
        num_dustbins: int = 2,
        **kwargs,
    ):
        # Shape-preserving path: we operate directly on the raw
        # ``[B, T, K, D]`` slot tensor; the base ``to_canonical_dims``
        # reshape would collapse the time axis and break the per-frame
        # LSAP solve.
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        if lambda_event < 0:
            raise ValueError(f"`lambda_event` must be ≥ 0, got {lambda_event}.")
        if not (0.0 <= c_bin <= 2.0):
            raise ValueError(
                f"`c_bin` should lie in the cosine-distance range [0, 2], got {c_bin}."
            )
        if num_dustbins not in (2, 4):
            raise ValueError(f"`num_dustbins` must be 2 or 4, got {num_dustbins}.")
        self.mask_key = mask_key
        self.lambda_event = float(lambda_event)
        self.c_bin = float(c_bin)
        self.num_dustbins = int(num_dustbins)
        # Stash per-step components for logging (set inside `forward`).
        self._last_components: Dict[str, torch.Tensor] = {}

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        # Bypass base class dim-check / canonicalization; we need the raw
        # ``[B, T, K, D]`` slot tensor.
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        # Targets are synthesized inside ``forward`` from
        # ``replay_target_trajectories`` (lifted via ``aux_keys``); no
        # separate target tensor is needed.
        return None

    @staticmethod
    def _require(name: str, value: Any) -> Any:
        """Hard-raise on missing aux tensor — no silent fallback (project rule)."""
        if value is None:
            raise ValueError(
                f"GSRSIdentityLoss requires aux tensor `{name}` in outputs; got None. "
                "Either plumb the replay branch through `models.forward` (it lifts "
                "`replay_target_trajectories`, `replay_event_flags`, `replay_family`, "
                "and `open_set_probs` from `inputs` / head output) or remove this "
                "loss from the config."
            )
        return value

    def forward(
        self,
        pred: torch.Tensor,
        target: Any = None,
        existence_mask: Optional[torch.Tensor] = None,
        replay_target_trajectories: Optional[torch.Tensor] = None,
        replay_event_flags: Optional[torch.Tensor] = None,
        replay_family: Optional[List[str]] = None,
        open_set_probs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the rectangular dustbin LSAP loss + event CE.

        Args:
            pred: ``[B, T, K_pred, D]`` student slot state.
            target: unused; kept for dispatch-signature compatibility.
            existence_mask: unused in the core computation; accepted for
                dispatch-signature parity.
            replay_target_trajectories: ``[B, T, K_target + K_dustbin, D]``
                target slot trajectories from the renderer; the last
                ``num_dustbins`` rows are dustbin virtuals. Required.
            replay_event_flags: ``[B, T, K_target + K_dustbin]`` int labels
                (indices into ``EVENT_TO_INDEX``). Required.
            replay_family: length-``B`` list of intervention family strings.
                Required. Currently used only for accounting; the core
                assignment + event CE are family-agnostic (per §3.11).
            open_set_probs: ``[B, T, K_pred, 4]`` softmax output of
                ``OpenSetIdentityHead``. Required.

        Returns:
            Scalar ``L_total = L_assign + λ_event · L_event``.

        Raises:
            ValueError: if any required aux tensor is missing, or on any
                shape mismatch.
        """
        del target, existence_mask, kwargs  # consumed via aux dispatch.

        # Hard-raise on any missing aux tensor — no silent fallback.
        self._require("replay_target_trajectories", replay_target_trajectories)
        self._require("replay_event_flags", replay_event_flags)
        self._require("replay_family", replay_family)
        self._require("open_set_probs", open_set_probs)

        # Lazy import to keep the top-of-file dependency footprint small;
        # matches the pattern used elsewhere in this codebase (see
        # ``networks.HungarianPredictor`` for precedent).
        from scipy.optimize import linear_sum_assignment

        if pred.ndim != 4:
            raise ValueError(
                "GSRSIdentityLoss expects `pred` of shape [B, T, K_pred, D]; "
                f"got ndim={pred.ndim} shape={tuple(pred.shape)}."
            )
        B, T, K_pred, D = pred.shape

        if replay_target_trajectories.ndim != 4:
            raise ValueError(
                "`replay_target_trajectories` must have shape "
                "[B, T, K_target + K_dustbin, D]; "
                f"got {tuple(replay_target_trajectories.shape)}."
            )
        B_t, T_t, K_total, D_t = replay_target_trajectories.shape
        if (B_t, T_t) != (B, T):
            raise ValueError(
                "`replay_target_trajectories` leading dims "
                f"{(B_t, T_t)} != pred leading dims {(B, T)}."
            )
        if D_t != D:
            raise ValueError(
                "`replay_target_trajectories` feature dim "
                f"{D_t} != pred feature dim {D}."
            )
        if K_total < self.num_dustbins + 1:
            raise ValueError(
                f"`replay_target_trajectories` has K_total={K_total} rows but "
                f"num_dustbins={self.num_dustbins} expected; need at least one "
                "real target row."
            )
        K_target = K_total - self.num_dustbins

        if replay_event_flags.shape != (B, T, K_total):
            raise ValueError(
                "`replay_event_flags` must have shape "
                f"[B, T, K_total]={(B, T, K_total)}; "
                f"got {tuple(replay_event_flags.shape)}."
            )

        if open_set_probs.ndim != 4 or open_set_probs.shape[:3] != (B, T, K_pred):
            raise ValueError(
                "`open_set_probs` must have shape [B, T, K_pred, 4]="
                f"{(B, T, K_pred, 4)}; got {tuple(open_set_probs.shape)}."
            )
        if open_set_probs.shape[-1] != len(self.EVENT_TO_INDEX):
            raise ValueError(
                "`open_set_probs` last dim must match len(EVENT_TO_INDEX)="
                f"{len(self.EVENT_TO_INDEX)}; got {open_set_probs.shape[-1]}."
            )

        if not isinstance(replay_family, (list, tuple)) or len(replay_family) != B:
            raise ValueError(
                f"`replay_family` must be a list/tuple of length B={B}; "
                f"got type={type(replay_family).__name__} len={len(replay_family) if hasattr(replay_family, '__len__') else 'N/A'}."
            )

        # --- Cost matrix: cosine distance between L2-normalised features.
        # Real rows (j < K_target) get cosine cost; dustbin rows (j ≥ K_target)
        # get the fixed scalar ``c_bin``. Cost is computed per (b, t) over
        # the ``[K_pred, K_total]`` rectangular matrix.
        pred_n = nn.functional.normalize(pred, p=2.0, dim=-1)                    # [B, T, K_pred, D]
        tgt_real = replay_target_trajectories[:, :, :K_target]                   # [B, T, K_target, D]
        tgt_real_n = nn.functional.normalize(tgt_real, p=2.0, dim=-1)            # [B, T, K_target, D]
        # Cosine similarity per (b, t, i_pred, j_target_real).
        cos_sim_real = torch.einsum("btid,btjd->btij", pred_n, tgt_real_n)       # [B, T, K_pred, K_target]
        cost_real = 1.0 - cos_sim_real                                            # cosine distance ∈ [0, 2]

        # Build the full rectangular cost matrix with dustbin columns.
        dustbin_block = cost_real.new_full(
            (B, T, K_pred, self.num_dustbins), fill_value=self.c_bin
        )
        cost_full = torch.cat([cost_real, dustbin_block], dim=-1)                # [B, T, K_pred, K_total]

        # --- Per-frame rectangular LSAP solve.
        # scipy's ``linear_sum_assignment`` handles rectangular matrices: if
        # K_pred != K_total, only the smaller side is fully matched. We solve
        # on the detached CPU numpy version of the cost (assignment is a
        # frozen training signal — no grads through the argmin).
        cost_np = cost_full.detach().cpu().numpy()

        # Collect matched (row, col) per (b, t) and accumulate losses.
        # ``L_assign`` is averaged over real-slot matches ONLY (j < K_target);
        # dustbin matches carry no gradient signal (their cost is fixed).
        # ``L_event`` is CE on the same matched pairs using the labels in
        # ``replay_event_flags`` at the matched target column.
        assign_num = cost_full.new_zeros(())       # scalar accumulator
        assign_denom = cost_full.new_zeros(())     # scalar count
        event_num = cost_full.new_zeros(())        # scalar accumulator
        event_denom = cost_full.new_zeros(())      # scalar count

        # Iterate over (b, t) frames. B and T are typically small (≤ 64);
        # LSAP dominates wall-clock, not the python loop.
        for b in range(B):
            for t in range(T):
                row_ind, col_ind = linear_sum_assignment(cost_np[b, t])
                if len(row_ind) == 0:
                    continue
                rows_tensor = torch.as_tensor(row_ind, device=pred.device, dtype=torch.long)
                cols_tensor = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)

                # --- Assignment CE (differentiable via the cosine cost).
                # Take the gradient-carrying cos similarity at matched pairs
                # that land on real target rows. Dustbin pairs have fixed
                # cost → skip (no gradient through the fixed scalar).
                real_mask = cols_tensor < K_target
                if real_mask.any():
                    rows_real = rows_tensor[real_mask]
                    cols_real = cols_tensor[real_mask]
                    sims_real = cos_sim_real[b, t, rows_real, cols_real]        # [n_real_matches]
                    # Assignment loss per spec: mean of (1 - cos_sim) over
                    # matched real pairs. Kept as a sum here; averaged below.
                    assign_num = assign_num + (1.0 - sims_real).sum()
                    assign_denom = assign_denom + float(real_mask.sum().item())

                # --- Event CE (through `open_set_probs[b, t, rows, :]`).
                # Target label per matched pair is `replay_event_flags[b, t, col]`.
                # CE is computed in log-prob form since the head already applies
                # softmax; we clamp for numerical safety, matching the pattern
                # in ``MaskIoUQualityLoss`` above.
                if len(rows_tensor) > 0:
                    probs_matched = open_set_probs[b, t, rows_tensor]            # [n_matches, 4]
                    labels_matched = replay_event_flags[b, t, cols_tensor].long()
                    if (labels_matched < 0).any() or (labels_matched >= open_set_probs.shape[-1]).any():
                        raise ValueError(
                            "`replay_event_flags` contains out-of-range class indices; "
                            f"expected [0, {open_set_probs.shape[-1]}), got "
                            f"min={labels_matched.min().item()} max={labels_matched.max().item()}."
                        )
                    log_probs = torch.log(probs_matched.clamp(min=1e-8))        # [n_matches, 4]
                    ce = -log_probs.gather(-1, labels_matched.unsqueeze(-1)).squeeze(-1)  # [n_matches]
                    event_num = event_num + ce.sum()
                    event_denom = event_denom + float(ce.numel())

        # --- Final aggregation. Guard against empty-match edge cases by
        # returning zero-with-grad rather than NaN.
        if assign_denom > 0:
            l_assign = assign_num / assign_denom
        else:
            l_assign = cost_full.sum() * 0.0

        if event_denom > 0:
            l_event = event_num / event_denom
        else:
            l_event = open_set_probs.sum() * 0.0

        l_total = l_assign + self.lambda_event * l_event

        # Stash components for logging without changing the dispatch
        # signature (ObjectCentricModel.compute_loss only reduces the
        # single returned scalar; individual components are read off
        # ``self._last_components`` by any upstream logger that chooses to).
        self._last_components = {
            "L_assign": l_assign.detach(),
            "L_event": l_event.detach(),
            "L_total": l_total.detach(),
        }

        return l_total


class DIOCReconstructionLoss(Loss):
    """Diffusion-Inpainted Occlusion Curriculum reconstruction loss (Idea A3).

    The DIOC data pipeline (:mod:`slotcontrast.data.dioc_occluder`) replaces
    the pixels of 1-2 slots' regions in the training video with diffusion-
    inpainted background. The student model sees the *occluded* video and is
    trained to still reconstruct the *original* (non-occluded) DINOv2 patch
    features. This forces slots to encode occlusion-invariant identity: the
    student cannot rely on the occluded object being visible in the input.

    Mechanism
    ---------
    - ``pred`` is ``decoder.reconstruction`` from the student's forward on the
      occluded video. Shape ``[B, T, P, D_feat]`` (matches
      :class:`MSELoss` in the base config).
    - ``original_video`` (aux key) is the un-occluded ``[B, T, 3, H, W]``
      video carried through by the occluder. We extract DINOv2 patch features
      of it on-the-fly with the student's own (frozen) backbone; matching the
      original ``loss_featrec`` target path exactly.
    - MSE between the two.
    - Only accumulate loss on the frames/pixels that belong to
      ``occluded_sample_idx`` — the rest of the batch contributes zero
      (multiplied by 0.0 with grad preserved, following the convention used
      in :class:`SlotDropRecoverLoss`).

    The target feature extraction is done by the *same* feature extractor the
    student uses for its standard ``loss_featrec`` target; we read it from the
    ``outputs`` dict (key ``target_features`` if already computed) or,
    failing that, hard-raise. Re-running DINOv2 on the original video would
    double backbone cost; the collator is expected to have stashed these
    features at occlusion time.

    Aux keys
    --------
    - ``original_video_features``: ``[B, T, P, D_feat]`` pre-computed DINOv2
      patch features of the un-occluded video. The collator computes these
      once per batch using the frozen teacher's encoder (same model used for
      mask extraction) and lifts them into ``outputs`` via the data-pipeline
      hook in :mod:`slotcontrast.models`.
    - ``occluded_sample_idx``: ``LongTensor[n_occluded]``. Samples not in
      this set contribute 0 to the loss.

    Weight
    ------
    Default ``lambda_dioc = 0.1`` set at config level (``loss_weights``).

    Raises
    ------
    ValueError: ``original_video_features`` shape does not match ``pred``.
    RuntimeError: Neither ``original_video_features`` nor the fallback path
        is available. (We do NOT fall back to re-encoding on the fly — that
        would silently double the backbone compute budget.)
    """

    aux_keys = ["original_video_features", "occluded_sample_idx"]

    def __init__(
        self,
        pred_key: str = "decoder.reconstruction",
        target_key: str = "dummy",
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        # Preserve the raw ``[B, T, P, D]`` shape so we can align with the
        # DINOv2 feature tensor directly.
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        self.mask_key = mask_key

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        # Target is the pre-computed original-video features carried in
        # ``outputs['original_video_features']``; ``forward`` pulls it via
        # the aux_keys dispatch.
        return None

    def forward(
        self,
        pred,
        target=None,
        existence_mask=None,
        original_video_features=None,
        occluded_sample_idx=None,
        **kwargs,
    ):
        """Compute the DIOC reconstruction loss.

        Args:
            pred: ``[B, T, P, D_feat]`` student decoder reconstruction on the
                occluded video. Must be in the feature-channel slice (the
                base config applies ``pred_dims=(0, FEAT_DIM)`` for
                ``loss_featrec``; the DIOC config mirrors that).
            target: unused; kept for dispatch-signature compatibility.
            existence_mask: unused (feature-level loss, not slot-level).
            original_video_features: ``[B, T, P, D_feat]`` DINOv2 patch
                features of the *un-occluded* video, detached.
            occluded_sample_idx: ``LongTensor[n_occluded]`` of batch indices
                where occlusion was applied. Only these samples contribute.

        Returns:
            Scalar loss tensor.
        """
        if original_video_features is None:
            raise RuntimeError(
                "DIOCReconstructionLoss: missing `original_video_features` "
                "aux kwarg. The data pipeline must pre-compute DINOv2 patch "
                "features of the un-occluded video and lift them into "
                "`outputs['original_video_features']` — no on-the-fly "
                "re-encoding fallback (would double backbone cost)."
            )
        target_feat = original_video_features.detach()

        # Align predicted feature channels with the target.
        if self.pred_dims is not None:
            # ``pred_dims`` is already applied by base.get_prediction when the
            # default path is used; but we bypass that in get_prediction() to
            # keep shapes. Apply the slice here.
            pred = pred[..., self.pred_dims]
        if pred.shape != target_feat.shape:
            raise ValueError(
                f"DIOCReconstructionLoss: pred shape {tuple(pred.shape)} "
                f"does not match original_video_features "
                f"{tuple(target_feat.shape)}."
            )

        B = pred.shape[0]
        if occluded_sample_idx is None or occluded_sample_idx.numel() == 0:
            # No occluded samples this batch: return zero with grad preserved.
            return pred.sum() * 0.0

        idx = occluded_sample_idx.to(device=pred.device, dtype=torch.long)
        pred_sel = pred[idx]
        target_sel = target_feat[idx]
        loss = torch.nn.functional.mse_loss(pred_sel, target_sel, reduction="mean")
        return loss


# ---------------------------------------------------------------------------
# Amodal Persistence Particles (APP) losses
# Reference: refine-logs/FINAL_PROPOSAL_2026_04_17.md §3.4
# ---------------------------------------------------------------------------


class AmodalSubsetLoss(Loss):
    """L_subset: enforces the invariant v ≤ a at every patch.

    penalty = mean( ReLU(v - a) )
    """

    def __init__(
        self,
        pred_key: str = "processor.visible_masks",
        target_key: str = "processor.amodal_masks",
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.target_path)

    def forward(self, v: torch.Tensor, a: torch.Tensor, **kwargs) -> torch.Tensor:
        violation = torch.relu(v - a)
        return violation.mean()


class AntiDilationLoss(Loss):
    """L_gap: penalizes hidden mass not justified by occlusion evidence.

    L_gap = mean( (a - v) * (1 - o) )

    ``o`` (occlusion plausibility) is routed via aux_keys.
    """

    aux_keys = ["occlusion_plausibility"]

    def __init__(
        self,
        pred_key: str = "processor.amodal_masks",
        target_key: str = "processor.visible_masks",
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.target_path)

    def forward(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        occlusion_plausibility: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden = (a - v).clamp(min=0)
        if occlusion_plausibility is not None:
            unjustified = hidden * (1.0 - occlusion_plausibility)
        else:
            unjustified = hidden
        return unjustified.mean()


class AmodalTransportLoss(Loss):
    """L_amodal: amodal field stability through occlusion.

    L = mean( |a - sg(ā)| * ω^occ )

    where ā is the transported amodal (stop-gradient) and ω^occ is an
    occlusion-conditioned weight (higher weight on patches where occlusion
    is plausible, since those are the ones the amodal field must preserve).
    """

    aux_keys = ["amodal_transported", "occlusion_plausibility"]

    def __init__(
        self,
        pred_key: str = "processor.amodal_masks",
        target_key: str = "dummy",
        occ_weight: float = 2.0,
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        self.occ_weight = occ_weight

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return None

    def forward(
        self,
        a: torch.Tensor,
        target: Any = None,
        amodal_transported: Optional[torch.Tensor] = None,
        occlusion_plausibility: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if amodal_transported is None:
            return a.sum() * 0.0
        a_bar = amodal_transported.detach()
        diff = (a - a_bar).abs()
        if occlusion_plausibility is not None:
            weight = 1.0 + (self.occ_weight - 1.0) * occlusion_plausibility.detach()
        else:
            weight = 1.0
        return (diff * weight).mean()


class ReappearanceConsistencyLoss(Loss):
    """L_reapp: InfoNCE over dormant particles for re-identification.

    For each dormant particle that reactivates, the positive is its own
    transported amodal support; negatives are the other particles' supports.
    This forces the amodal field to carry identity-discriminative information
    through occlusion.

    Operates on per-frame outputs stacked by ScanOverTime.
    """

    aux_keys = ["amodal_transported", "particle_states", "prev_particle_states"]

    def __init__(
        self,
        pred_key: str = "processor.visible_masks",
        target_key: str = "dummy",
        temperature: float = 0.1,
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )
        self.temperature = temperature

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return None

    def forward(
        self,
        v: torch.Tensor,
        target: Any = None,
        amodal_transported: Optional[torch.Tensor] = None,
        particle_states: Optional[torch.Tensor] = None,
        prev_particle_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if amodal_transported is None or particle_states is None:
            return v.sum() * 0.0

        a_bar = amodal_transported.detach()

        if v.ndim == 4:
            B, T, K, P = v.shape
            v_flat = v.reshape(B * T, K, P)
            a_bar_flat = a_bar.reshape(B * T, K, P)
            m_flat = particle_states.reshape(B * T, K)
            prev_m_flat = prev_particle_states.reshape(B * T, K) if prev_particle_states is not None else m_flat
        else:
            v_flat = v
            a_bar_flat = a_bar
            m_flat = particle_states
            prev_m_flat = prev_particle_states if prev_particle_states is not None else m_flat

        BT, K, P = v_flat.shape
        if K <= 1:
            return v.sum() * 0.0

        v_n = nn.functional.normalize(v_flat, p=2.0, dim=-1)
        a_n = nn.functional.normalize(a_bar_flat, p=2.0, dim=-1)

        sim = torch.bmm(v_n, a_n.transpose(1, 2)) / self.temperature

        # Target: particles that were dormant last frame (candidates for
        # reappearance). This captures both successful reactivations and
        # particles still dormant (encouraging their amodal field to stay
        # discriminative for future matching).
        was_dormant = (prev_m_flat == 1)
        if not was_dormant.any():
            return v.sum() * 0.0

        labels = torch.arange(K, device=v.device).unsqueeze(0).expand(BT, K)
        loss_per = nn.functional.cross_entropy(
            sim.view(BT * K, K), labels.reshape(BT * K), reduction="none"
        ).view(BT, K)

        valid_count = was_dormant.float().sum()
        if valid_count > 0:
            loss = (loss_per * was_dormant.float()).sum() / valid_count
        else:
            loss = v.sum() * 0.0
        return loss


class SupportConservationLoss(Loss):
    """L_support: weak mass-stability regularizer.

    Penalizes large frame-to-frame changes in total amodal support mass
    so that support does not explode or vanish between birth/death events.

    L = mean( |s_t - s_{t-1}| )  where s = ||a||_1
    """

    def __init__(
        self,
        pred_key: str = "processor.amodal_masks",
        target_key: str = "dummy",
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key, target_key,
            video_inputs=False, patch_inputs=False, keep_input_dim=True,
            **kwargs,
        )

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return None

    def forward(self, a: torch.Tensor, target: Any = None, **kwargs) -> torch.Tensor:
        if a.ndim == 4:
            B, T, K, P = a.shape
            support = a.sum(dim=-1)
            if T < 2:
                return a.sum() * 0.0
            delta = (support[:, 1:] - support[:, :-1]).abs()
            return delta.mean()
        return a.sum() * 0.0


# --- D3 MOCSP (Round 5 Phase C1) -----------------------------------------

class MOCSPLoss(Loss):
    """D3 Masked Object-Conditioned Self-Prediction loss.

    Per-slot held-out feature regression against a frozen DINO target.
    When paired with ``MaskingFrameEncoder``, the system implements a
    MAE-style bottleneck: slot attention consumes a partially-masked
    feature map (``mask_ratio`` of patches are replaced with a learnable
    mask token), and this loss supervises each slot's per-patch decoder
    output to predict the *unmasked* target features at held-out positions.

    When paired with the standard ``FrameEncoder`` (no input masking),
    MOCSP reduces to a stochastic per-slot frozen-target regression with
    support weighting — a cleaner ablation that keeps the slot pipeline
    unchanged but turns featrec into a per-slot objective.

    Design notes (addressed in Codex reviews 2026-04-23):
      * ``masks_soft.detach()`` on the weight path: MOCSP does NOT compete
        with the primary ``MSELoss`` featrec for decoder-mask-head gradient.
        The mask head is trained by featrec; MOCSP only weights its own
        per-slot regression by the (stop-grad) soft support.
      * ``mean`` reduction over the feature dim D so the scalar magnitude
        is comparable to ``torch.nn.MSELoss(reduction="mean")``.
      * Hard errors on missing target / wrong prediction shape: the loss
        is safe to wire but refuses to silently no-op.

    Args:
        pred_key: where to read per-slot decoder output; the default
            assumes ``MOCSPMLPDecoder`` is used (exposes ``recons_per_slot``).
        target_key: frozen-target features; defaults to the R2 pathway
            ``target_encoder.backbone_features``.
        mask_ratio: hold-out ratio on the (slot, patch) grid.
        mask_mode: one of ``{"input_mask","slot_conditioned","random","none"}``.
            - ``input_mask``: use the encoder-emitted ``input_mask`` if present
              (MAE-style input masking → loss only on held-out patches).
            - ``slot_conditioned``: independently sample per slot.
            - ``random``: sample once per patch, broadcast across slots.
            - ``none``: no hold-out (pure per-slot support-weighted MSE).
        soft_support: weight the per-slot MSE by ``decoder_masks_soft``
            (stop-grad).
        confidence_floor: lower bound on the soft support weight (avoids
            zero-weight edge cases when mask head hasn't warmed up).
        normalize_features: L2-normalise pred/target along D before MSE
            (matches DINO-style cosine-proximity semantics; off by default).
        strict_require_target: hard-error when the target feature is missing.
    """

    def __init__(
        self,
        pred_key: str = "decoder.recons_per_slot",
        target_key: str = "target_encoder.backbone_features",
        mask_ratio: float = 0.5,
        mask_mode: str = "input_mask",
        soft_support: bool = True,
        confidence_floor: float = 0.05,
        eps: float = 1e-6,
        normalize_features: bool = False,
        strict_require_target: bool = True,
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        kwargs.pop("video_inputs", None)
        kwargs.pop("patch_inputs", None)
        kwargs.pop("keep_input_dim", None)
        super().__init__(
            pred_key,
            target_key,
            video_inputs=True,
            patch_inputs=True,
            keep_input_dim=True,
            **kwargs,
        )
        valid_modes = ("input_mask", "slot_conditioned", "random", "none")
        if mask_mode not in valid_modes:
            raise ValueError(
                f"MOCSPLoss.mask_mode must be one of {valid_modes}, got {mask_mode!r}"
            )
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"MOCSPLoss.mask_ratio must be in [0, 1], got {mask_ratio}"
            )
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = mask_mode
        self.soft_support = bool(soft_support)
        self.confidence_floor = float(confidence_floor)
        self.eps = float(eps)
        self.normalize_features = bool(normalize_features)
        self.strict_require_target = bool(strict_require_target)
        # Route existence_mask via mask_key; route decoder_masks_soft +
        # input_mask via aux_keys. See models.py:compute_loss.
        self.mask_key = mask_key
        self.aux_keys = ["decoder_masks_soft", "input_mask"]
        # diagnostics
        self._last_mask_coverage: Optional[torch.Tensor] = None
        self._last_effective_weight: Optional[torch.Tensor] = None
        self._last_per_slot_loss_mean: Optional[torch.Tensor] = None

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        # Custom prediction path: recons_per_slot is 5D and bypasses the
        # base canonicalizer (which only handles up to 4D tensors).
        pred = utils.read_path(outputs, elements=self.pred_path)
        if self.pred_dims is not None:
            pred = pred[..., self.pred_dims]
        return pred

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path, error=False)
        if target is None:
            return None
        return target.detach()

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor],
        existence_mask: Optional[torch.Tensor] = None,
        decoder_masks_soft: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if target is None:
            if self.strict_require_target:
                raise RuntimeError(
                    "MOCSPLoss target is None. MOCSP requires "
                    "`target_encoder.backbone_features` (set "
                    "`model.target_encoder_input: video` with a frozen "
                    "target_encoder; see "
                    "configs/slotcontrast/app/movi_d_gcv1_pilot15k_r2_only.yaml)."
                )
            return pred.new_zeros(())
        if pred.ndim != 5:
            raise RuntimeError(
                f"MOCSPLoss expects pred=[B,T,K,P,D]; got shape "
                f"{tuple(pred.shape)}. Use MOCSPMLPDecoder so "
                f"`decoder.recons_per_slot` is emitted."
            )
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != 4:
            raise RuntimeError(
                f"MOCSPLoss expects target=[B,(T,)P,D]; got shape "
                f"{tuple(target.shape)}"
            )

        B, T, K, P, D = pred.shape
        Bt, Tt, Pt, Dt = target.shape
        if not (B == Bt and P == Pt and D == Dt):
            raise RuntimeError(
                f"MOCSPLoss shape mismatch pred={tuple(pred.shape)} vs "
                f"target={tuple(target.shape)}"
            )
        if Tt == 1 and T > 1:
            target = target.expand(B, T, P, D)
        elif Tt != T:
            raise RuntimeError(
                f"MOCSPLoss time mismatch pred_T={T} vs target_T={Tt}"
            )

        device = pred.device

        # ---- mask sampling -----------------------------------------------
        if self.mask_mode == "input_mask":
            # Use encoder-emitted input_mask [B, T, P] (MAE-style).
            if input_mask is None:
                if self.strict_require_target:
                    raise RuntimeError(
                        "MOCSPLoss.mask_mode='input_mask' requires the encoder "
                        "to emit `input_mask` (use MaskingFrameEncoder). For a "
                        "loss-only variant, set mask_mode='slot_conditioned' or "
                        "'random'."
                    )
                m_shared = torch.ones(B, T, P, device=device)
            else:
                if input_mask.ndim == 2:
                    # [B, P] → [B, 1, P] → [B, T, P]
                    m_shared = input_mask.unsqueeze(1).expand(B, T, P).float()
                elif input_mask.ndim == 3:
                    m_shared = input_mask.float()
                else:
                    raise RuntimeError(
                        f"MOCSPLoss.input_mask expected [B,P] or [B,T,P], got "
                        f"shape {tuple(input_mask.shape)}"
                    )
                # Eval-time safety (Codex Round-5 review catch): if the
                # encoder ran without masking (e.g., validation with
                # ``mask_on_eval=False``) the emitted ``input_mask`` is
                # all-zero and ``weight.sum()`` would be 0 — yielding a
                # silently zero ``val/loss_mocsp``. Sample a local
                # stochastic mask in that case so the validation scalar
                # remains meaningful.
                if m_shared.sum() < self.eps:
                    m_shared = (
                        torch.rand(B, T, P, device=device) < self.mask_ratio
                    ).float()
            # Broadcast across slots
            m = m_shared.unsqueeze(2).expand(B, T, K, P)
        elif self.mask_mode == "slot_conditioned":
            m = (torch.rand(B, T, K, P, device=device) < self.mask_ratio).float()
        elif self.mask_mode == "random":
            m_shared = (torch.rand(B, T, 1, P, device=device) < self.mask_ratio).float()
            m = m_shared.expand(B, T, K, P)
        else:  # "none"
            m = torch.ones(B, T, K, P, device=device)

        # ---- per-element MSE normalised over D --------------------------
        pred_to_use = pred
        target_to_use = target.detach()
        if self.normalize_features:
            pred_to_use = torch.nn.functional.normalize(pred_to_use, dim=-1)
            target_to_use = torch.nn.functional.normalize(target_to_use, dim=-1)
        sqerr = ((pred_to_use - target_to_use.unsqueeze(2)) ** 2).mean(dim=-1)  # [B,T,K,P]

        # ---- weight: held-out × soft-support × existence -----------------
        if self.soft_support:
            if decoder_masks_soft is None:
                raise RuntimeError(
                    "MOCSPLoss.soft_support=True requires the decoder to "
                    "emit `decoder_masks_soft` at the top level of the "
                    "outputs dict (lifted by models.py from decoder.masks). "
                    "If the decoder produces no masks, set "
                    "`soft_support: false` in the config to fall back to "
                    "uniform per-slot weighting."
                )
            support = decoder_masks_soft.detach()
            support = torch.clamp(
                support, min=self.confidence_floor / float(K), max=1.0
            )
            weight = m * support
        else:
            weight = m
        if existence_mask is not None:
            if existence_mask.ndim == 2:
                existence_mask = existence_mask.unsqueeze(1).expand(B, T, K)
            weight = weight * existence_mask.unsqueeze(-1).float()

        num = (sqerr * weight).sum()
        den = weight.sum().clamp(min=self.eps)
        loss = num / den

        with torch.no_grad():
            self._last_mask_coverage = m.mean().detach()
            self._last_effective_weight = weight.mean().detach()
            # per-slot loss mean (for diagnostic): weight-normalised per-k
            per_slot_num = (sqerr * weight).sum(dim=(0, 1, 3))           # [K]
            per_slot_den = weight.sum(dim=(0, 1, 3)).clamp(min=self.eps)  # [K]
            self._last_per_slot_loss_mean = (per_slot_num / per_slot_den).detach()

        return loss


class VICRegFeatureLoss(Loss):
    """VICReg-style variance + covariance regularizer on encoder features.

    Inspired by Bardes et al. (VICReg, ICLR 2022). Goal in this project:
    PROACTIVELY prevent the LoRA-induced feature rank collapse documented by
    the W4 diagnostic. The regularizer has two terms:

      L_var = mean over feature dims d of max(0, gamma - sqrt(var(features_d))).
        - Penalises any dimension whose stddev across the batch falls below
          ``var_target``. This pushes per-dim variance up; it is the
          differentiable analogue of "keep the participation ratio high".

      L_cov = mean of off-diagonal entries of the (D, D) covariance matrix
        squared, normalised by D.
        - Penalises correlated dimensions, which is the other direction of
          rank collapse (features all pointing in the same subspace).

    Combined loss = ``var_weight`` * L_var + ``cov_weight`` * L_cov.

    The loss is computed on flattened (B*T*P, D) tokens — i.e., the same
    quantity as the W4 diagnostic, so the regularizer directly targets the
    statistic we observe collapsing.

    Args:
        pred_key: dotted path to the encoder feature tensor; expects shape
            (B, T, P, D) with ``video_inputs=True, patch_inputs=True``, or
            a 2D/3D variant if those flags differ.
        target_key: dummy; the loss has no target. Required by the base class.
        var_target: stddev floor (gamma in the paper). 1.0 is the standard
            VICReg default for L2-normalised features; for raw DINOv2 ViT
            outputs we default to a smaller floor to avoid pushing magnitudes
            to high values that fight featrec.
        var_weight, cov_weight: scalar multipliers.
        eps: numerical floor for sqrt.
        ramp_start_step: if > 0, the regularizer is gated to zero before this
            step. Used to align with R3 (delayed LoRA unfreeze) so the
            regularizer only kicks in once LoRA starts moving.
        ramp_steps: linear ramp duration after ``ramp_start_step``.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str = "dummy",
        var_target: float = 0.5,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        eps: float = 1e-4,
        ramp_start_step: int = 0,
        ramp_steps: int = 0,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.var_target = float(var_target)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.eps = float(eps)
        self.ramp_start_step = int(ramp_start_step)
        self.ramp_steps = int(ramp_steps)
        # Mutable global step buffer set by the model's training_step before
        # compute_loss. Buffer (not a parameter) so checkpointing is correct.
        self.register_buffer(
            "_global_step", torch.tensor(0, dtype=torch.long), persistent=False
        )

        # Diagnostics for training_step logging.
        self._last_var_loss = None
        self._last_cov_loss = None
        self._last_var_below_target_frac = None
        self._last_ramp_factor = None

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        # Loss has no target; return a dummy aligned with prediction shape so the
        # framework's downstream `prediction.shape == target.shape` assumptions
        # don't trip. Actual computation in `forward` ignores target.
        prediction = self.get_prediction(outputs)
        return torch.zeros_like(prediction).detach()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # prediction: canonicalised to (B, N, D) where N = T * P (or P).
        if prediction.ndim != 3:
            raise ValueError(
                f"VICRegFeatureLoss expects (B, N, D) prediction; got {prediction.shape}"
            )
        B, N, D = prediction.shape
        feats = prediction.reshape(B * N, D).float()
        feats = feats - feats.mean(dim=0, keepdim=True)

        # Variance term: hinge below var_target on per-dim stddev.
        std = torch.sqrt(feats.var(dim=0, unbiased=False) + self.eps)
        var_loss = torch.relu(self.var_target - std).mean()

        # Covariance term: ||off-diag(C)||^2 / D, with C = feats^T feats / (N-1).
        if feats.shape[0] > 1:
            cov = (feats.T @ feats) / float(feats.shape[0] - 1)
            off_diag_sq = cov.pow(2).sum() - cov.pow(2).diagonal().sum()
            cov_loss = off_diag_sq / float(D)
        else:
            cov_loss = feats.new_zeros(())

        # Ramp factor: 0 before ramp_start_step, linearly ramps to 1 over
        # ramp_steps. When ramp_start_step==0 and ramp_steps==0 the factor is
        # always 1, matching the unramped behaviour.
        step = int(self._global_step.item())
        if step < self.ramp_start_step:
            ramp = 0.0
        elif self.ramp_steps > 0:
            ramp = min(1.0, (step - self.ramp_start_step) / float(self.ramp_steps))
        else:
            ramp = 1.0

        loss = ramp * (self.var_weight * var_loss + self.cov_weight * cov_loss)

        with torch.no_grad():
            self._last_var_loss = var_loss.detach()
            self._last_cov_loss = cov_loss.detach()
            self._last_var_below_target_frac = (std < self.var_target).float().mean().detach()
            self._last_ramp_factor = ramp

        return loss
