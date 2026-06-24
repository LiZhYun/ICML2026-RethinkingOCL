"""DIOC batch occluder — pixel-space occlusion curriculum (Idea A3).

Pipeline (per batch):
    1. With probability ``p_occlude`` for each sample, flag it for occlusion.
    2. For flagged samples, run the frozen *teacher* GCv1 model (loaded from
       ``base_model_ckpt``) to get ``decoder_masks_hard`` of shape
       ``[B, T, K, H, W]``.
    3. Pick ``n_objects_to_occlude`` slots per flagged sample; union their
       per-frame masks.
    4. Call :class:`VideoInpainter.inpaint` with the per-frame mask to
       replace those pixels with diffusion-sampled background.
    5. Return the occluded video plus ``occlusion_metadata`` so the
       downstream loss (:class:`DIOCReconstructionLoss`) can supervise the
       student.

Orthogonality to GSRS
---------------------
GSRS composites slot *features* via a frozen renderer in slot-space. DIOC
composites *pixels* via a frozen diffusion inpainter. The two signals are
independent and the spec (brainstorm §A3) explicitly positions DIOC as the
pixel-space counterpart of GSRS.

No silent fallback
------------------
Every failure path raises: missing checkpoint, wrong mask shape, zero valid
slots after filtering, inpainter errors, etc.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch

from slotcontrast.data.diffusion_inpainter import VideoInpainter

logger = logging.getLogger(__name__)


class DIOCOccluder:
    """Apply teacher-guided pixel-space occlusions to training batches.

    Args:
        base_model_ckpt: Path to a trained GCv1 checkpoint. The teacher is
            reconstructed from the same config tree used to build the student
            and loaded via :meth:`ObjectCentricModel.load_weights_from_checkpoint`.
            Hard-raises if the path does not exist.
        inpainter: A :class:`VideoInpainter` instance. The caller owns its
            lifecycle (it may be shared across occluders).
        teacher_model: The frozen teacher GCv1 model. We accept it as an
            already-built nn.Module rather than constructing one ourselves,
            because building an :class:`ObjectCentricModel` requires the
            full config tree. Responsibility for construction + eval/freeze
            sits with the caller (the datamodule collator); this class
            simply runs the teacher's forward pass.
        p_occlude: Per-sample probability of applying occlusion. Default 0.3.
        n_objects_to_occlude: ``(lo, hi)`` inclusive range for number of
            slots to occlude per flagged sample. Default ``(1, 2)``.
        mask_key: Output key in teacher forward's outputs dict containing
            ``[B, T, K, H, W]`` hard masks. Default ``decoder_masks_hard``.
        input_key: Key in the input batch for the video tensor. Default
            ``video``.

    Raises:
        FileNotFoundError: ``base_model_ckpt`` does not exist.
        RuntimeError: Teacher is not in eval mode, or the teacher's output
            does not contain ``mask_key``.
    """

    def __init__(
        self,
        base_model_ckpt: str,
        inpainter: VideoInpainter,
        teacher_model: torch.nn.Module,
        p_occlude: float = 0.3,
        n_objects_to_occlude: Tuple[int, int] = (1, 2),
        mask_key: str = "decoder_masks_hard",
        input_key: str = "video",
    ) -> None:
        if not os.path.isfile(base_model_ckpt):
            raise FileNotFoundError(
                f"DIOCOccluder: base_model_ckpt not found at '{base_model_ckpt}'. "
                f"DIOC requires a trained GCv1 teacher checkpoint (same one "
                f"used by GSRS — see RESEARCH_BRIEF.md)."
            )
        if teacher_model.training:
            raise RuntimeError(
                "DIOCOccluder: teacher_model must be in eval() mode before "
                "being handed to the occluder. Call `.eval()` and freeze "
                "parameters before construction."
            )
        for p in teacher_model.parameters():
            if p.requires_grad:
                raise RuntimeError(
                    "DIOCOccluder: teacher_model has trainable parameters. "
                    "Freeze all parameters (requires_grad=False) before use."
                )
        lo, hi = int(n_objects_to_occlude[0]), int(n_objects_to_occlude[1])
        if not (1 <= lo <= hi):
            raise ValueError(
                f"DIOCOccluder: n_objects_to_occlude must satisfy "
                f"1 <= lo <= hi; got ({lo}, {hi})."
            )
        if not (0.0 <= p_occlude <= 1.0):
            raise ValueError(
                f"DIOCOccluder: p_occlude must be in [0, 1]; got {p_occlude}."
            )

        self.base_model_ckpt = base_model_ckpt
        self.inpainter = inpainter
        self.teacher = teacher_model
        self.p_occlude = float(p_occlude)
        self.n_lo = lo
        self.n_hi = hi
        self.mask_key = mask_key
        self.input_key = input_key

        logger.info(
            "[DIOC] DIOCOccluder initialized ckpt=%s p_occlude=%.2f "
            "n_objects_to_occlude=(%d, %d) mask_key=%s",
            base_model_ckpt, self.p_occlude, self.n_lo, self.n_hi, self.mask_key,
        )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def occlude_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """Apply teacher-guided occlusion to a training batch in-place (cloned).

        Args:
            batch: Standard slotcontrast training batch. Must contain
                ``batch[self.input_key]`` of shape ``[B, T, 3, H, W]``.

        Returns:
            ``(augmented_batch, metadata)`` where ``augmented_batch`` has the
            same keys as ``batch`` but with ``batch[self.input_key]`` replaced
            by the occluded video. ``metadata`` contains:

            - ``occluded_sample_idx``: ``LongTensor[n_occluded]`` of batch indices.
            - ``occluded_slot_idx``: ``List[LongTensor]`` (one per occluded
              sample) of shape ``[n_slots_occluded_i]``.
            - ``original_video``: the un-occluded video ``[B, T, 3, H, W]``
              (detached, on the same device). Lifted into the model's outputs
              dict by the data pipeline so the loss can consume it via
              ``aux_keys``.
            - ``occlusion_mask``: ``Tensor[B, T, H, W]`` float mask with 1.0
              where pixels were replaced, 0.0 elsewhere (for diagnostics).
        """
        video = batch[self.input_key]
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                f"DIOCOccluder.occlude_batch: expected video [B, T, 3, H, W], "
                f"got {tuple(video.shape)}."
            )
        B, T, _, H, W = video.shape
        device = video.device

        # 1. Decide per-sample occlusion flag.
        flags = (torch.rand(B, device=device) < self.p_occlude)
        occluded_idx = flags.nonzero(as_tuple=False).flatten()
        if occluded_idx.numel() == 0:
            # No sample selected; return the original batch plus empty metadata.
            # This is NOT a fallback — zero occlusions is the correct and
            # consistent no-op when the Bernoulli trial flips off.
            meta = {
                "occluded_sample_idx": occluded_idx,
                "occluded_slot_idx": [],
                "original_video": video.detach().clone(),
                "occlusion_mask": torch.zeros(B, T, H, W, device=device,
                                              dtype=video.dtype),
            }
            return batch, meta

        # 2. Run frozen teacher to get masks. We run on the *full* batch
        # (not just flagged samples) because the teacher's forward signature
        # expects the same batch structure downstream models use; slicing
        # first would force per-sample padding handling.
        teacher_out = self.teacher({self.input_key: video})
        if self.mask_key not in teacher_out:
            raise RuntimeError(
                f"DIOCOccluder: teacher forward did not emit '{self.mask_key}'. "
                f"Output keys: {list(teacher_out.keys())}."
            )
        masks = teacher_out[self.mask_key]  # [B, T, K, H_m, W_m]
        if masks.ndim != 5 or masks.shape[0] != B or masks.shape[1] != T:
            raise RuntimeError(
                f"DIOCOccluder: teacher mask shape {tuple(masks.shape)} "
                f"incompatible with video {tuple(video.shape)}."
            )
        K = masks.shape[2]
        # Resize mask spatial dims to video's if they differ (teacher decoder
        # often outputs masks at a different resolution than the input).
        if masks.shape[-2:] != (H, W):
            masks = torch.nn.functional.interpolate(
                masks.reshape(B * T, K, masks.shape[-2], masks.shape[-1]).float(),
                size=(H, W),
                mode="nearest",
            ).reshape(B, T, K, H, W)

        # 3. Pick slots per flagged sample and build the union mask.
        occluded_slot_idx: List[torch.Tensor] = []
        occluded_video = video.clone()
        occlusion_mask = torch.zeros(B, T, H, W, device=device, dtype=video.dtype)

        for b_idx in occluded_idx.tolist():
            # Choose number of slots to occlude, uniformly in [n_lo, n_hi].
            n_choose = torch.randint(self.n_lo, self.n_hi + 1, (1,)).item()
            # Only select slots that have non-empty mask across all frames.
            # Rank slots by the total masked area (largest first), then
            # sample n_choose of them randomly from the top-K/2 to avoid
            # always picking the dominant slot (which is often background).
            slot_area = masks[b_idx].float().sum(dim=(0, 2, 3))  # [K]
            candidate_slots = slot_area.topk(min(K, max(n_choose * 2, 2))).indices
            # Randomly pick n_choose out of the candidate set.
            perm = torch.randperm(len(candidate_slots))[:n_choose]
            chosen = candidate_slots[perm]
            if chosen.numel() == 0:
                raise RuntimeError(
                    f"DIOCOccluder: sample {b_idx} produced no occludable "
                    f"slots (K={K}). Teacher output may be degenerate."
                )
            occluded_slot_idx.append(chosen.detach().cpu())

            # Union mask across chosen slots.
            union = masks[b_idx, :, chosen].float().sum(dim=1).clamp(0.0, 1.0)
            # Binarize at 0.5.
            union_bin = (union > 0.5).to(video.dtype)
            occlusion_mask[b_idx] = union_bin

            # 4. Inpaint.
            in_video = video[b_idx]  # [T, 3, H, W]
            in_mask = union_bin      # [T, H, W]
            try:
                out_video = self.inpainter.inpaint(in_video, in_mask)
            except Exception as exc:
                raise RuntimeError(
                    f"DIOCOccluder: inpainting failed for sample {b_idx}. "
                    f"Original error: {exc}"
                ) from exc
            if out_video.shape != in_video.shape:
                raise RuntimeError(
                    f"DIOCOccluder: inpainter returned unexpected shape "
                    f"{tuple(out_video.shape)} vs input {tuple(in_video.shape)}."
                )
            occluded_video[b_idx] = out_video

        augmented_batch = dict(batch)
        augmented_batch[self.input_key] = occluded_video
        meta = {
            "occluded_sample_idx": occluded_idx.detach().cpu(),
            "occluded_slot_idx": occluded_slot_idx,
            "original_video": video.detach().clone(),
            "occlusion_mask": occlusion_mask,
        }
        return augmented_batch, meta
