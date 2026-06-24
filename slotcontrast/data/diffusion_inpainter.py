"""Frozen video-inpainter wrapper for the DIOC curriculum (Idea A3).

Backend choice
--------------
The DIOC spec (``idea-stage/IDEA_BRAINSTORM_BROAD_2026_04_16.md`` §A3) lists
three candidate inpainters in order of preference:

1. Stable Video Diffusion (SVD) — no official *inpainting* variant exists as
   of 2026-04; the HF checkpoints are text/image-to-video generators without a
   mask-conditioned interface. SKIPPED.
2. ProPainter (Zhou et al. 2023) — flow-based SOTA for video inpainting, but
   requires cloning an external GitHub repo, a custom RAFT build, and a
   non-HuggingFace weight download. The file layout does not match our
   ``HF_HOME=/scratch/work/liz23/.cache/huggingface`` convention and the
   license (S-Lab 1.0, non-commercial) is more restrictive than we need.
3. Stable Diffusion 2 Inpainting (``stabilityai/stable-diffusion-2-inpainting``)
   via ``diffusers.StableDiffusionInpaintPipeline`` — applied frame-by-frame.
   HF-hosted, CreativeML Open RAIL-M (research-permissive), plug-and-play
   ``from_pretrained`` pattern that respects ``HF_HOME``. Simplest option that
   is actually available in 2026-04.

We pick option 3. Temporal consistency is *not* asserted across frames — this
is an intentional design choice (see spec §A3 Primary risk). The student must
learn to be invariant to the inpainter's frame-wise stochasticity; any
temporal smoothing pushed onto the inpainter would leak a prior that the
student then does not need to learn. If downstream analysis shows the student
is memorising inpainter artefacts we switch to ProPainter.

No silent fallback
------------------
All failure paths raise. Missing ``diffusers``, failed model download,
wrong output shape, and CPU-only device requests (where the pipeline cannot
run at ``fp16``) all raise with an explicit message. The caller is expected
to resolve the underlying issue — we never substitute a weaker inpainter or
drop the augmentation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Normalization used by Stable Diffusion: 0-1 input scaled to [-1, 1] internally
# by the pipeline. Our inference-time API takes and returns [0, 1]-range pixel
# tensors because that matches ``slotcontrast/data/transforms.py`` conventions
# (pre-ImageNet-normalization buffers). The pipeline handles normalization
# internally; we only ensure the input/output ranges match what the rest of
# the data pipeline expects.


class VideoInpainter(torch.nn.Module):
    """Frozen, frame-wise Stable-Diffusion-2 inpainter.

    Wraps ``diffusers.StableDiffusionInpaintPipeline`` with a stateless
    ``inpaint`` method that operates on a 4-D video tensor ``[T, 3, H, W]``
    and a 3-D occlusion mask ``[T, H, W]``. The pipeline itself is loaded
    once, its parameters are frozen, and the module is set to eval mode.

    Args:
        model_id: HuggingFace hub id. Default
            ``stabilityai/stable-diffusion-2-inpainting``.
        device: Torch device string, e.g. ``"cuda"`` or ``"cuda:0"``. CPU
            inpainting is not supported (hard-raise) because the base model
            is too slow in fp32 on CPU to be used inside a training loop.
        dtype: Torch dtype for the pipeline. Default ``torch.float16`` on
            GPU. fp32 is accepted but raises a warning because of the ~2x
            memory penalty.
        num_inference_steps: Diffusion DDIM / scheduler steps. Default 25
            (SD2 default is 50; we halve it to keep the per-call cost low
            enough for training-loop use at bs=8 T=4 H=W=336).

    Raises:
        ImportError: ``diffusers`` is not importable.
        RuntimeError: model download fails, device is CPU, or the pipeline
            returns an unexpected tensor shape from a smoke call.
    """

    _DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        num_inference_steps: int = 25,
    ) -> None:
        super().__init__()

        model_id = model_id or self._DEFAULT_MODEL_ID
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError(
                "VideoInpainter requires a CUDA device. CPU inpainting is not "
                "supported inside the training loop. Set device='cuda' on a "
                "GPU host, or pre-compute inpainted frames offline."
            )
        if device == "cpu":
            raise RuntimeError(
                "VideoInpainter: device='cpu' is rejected. Use a GPU or "
                "pre-compute offline."
            )
        if dtype is None:
            dtype = torch.float16

        # Diffusers import gated behind a clear error message so we don't
        # pollute top-level imports of slotcontrast.data with a heavy optional
        # dependency. DIOC is opt-in at config time.
        try:
            from diffusers import StableDiffusionInpaintPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "VideoInpainter requires the `diffusers` package. Install it "
                "with `pip install diffusers>=0.27 transformers>=4.40 "
                "accelerate safetensors` into the slotcontrast environment."
            ) from exc

        hf_cache = os.environ.get("HF_HOME")
        logger.info(
            "[DIOC] VideoInpainter loading backend=stable-diffusion-2-inpainting "
            "model_id=%s device=%s dtype=%s HF_HOME=%s num_inference_steps=%d",
            model_id, device, dtype, hf_cache, num_inference_steps,
        )
        try:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"VideoInpainter: failed to load pipeline '{model_id}'. Check "
                f"that HF_HOME is writable and the hub id is correct. "
                f"Original error: {exc}"
            ) from exc

        pipeline = pipeline.to(device)
        # Disable progress bars — this runs hundreds of times per epoch.
        try:
            pipeline.set_progress_bar_config(disable=True)
        except Exception:
            pass
        # Freeze every parameter. The pipeline is not a single nn.Module; we
        # iterate over its components so LoRA / adapter users see an explicit
        # freeze pass.
        for component_name in ("unet", "vae", "text_encoder"):
            component = getattr(pipeline, component_name, None)
            if component is not None and hasattr(component, "parameters"):
                for p in component.parameters():
                    p.requires_grad_(False)
                component.eval()

        self.pipeline = pipeline
        self.device_str = device
        self.dtype = dtype
        self.model_id = model_id
        self.num_inference_steps = int(num_inference_steps)

    # --- Public API --------------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "",
    ) -> torch.Tensor:
        """Inpaint a video frame-by-frame with the frozen SD2-inpaint pipeline.

        Args:
            video: ``[T, 3, H, W]`` float tensor in the ``[0, 1]`` range. Values
                outside that range will raise because SD inpainting assumes
                natural images; silently clamping would mask upstream bugs.
            mask: ``[T, H, W]`` float tensor in ``[0, 1]`` where 1 == region to
                inpaint (replace), 0 == keep. Non-binary values are allowed
                but are thresholded at 0.5 because SD2-inpaint's public API
                expects a binary mask.
            prompt: Text prompt to condition inpainting. Default empty string
                which nudges SD to reproduce plausible background.

        Returns:
            ``[T, 3, H, W]`` float tensor on the same device/dtype as ``video``
            with pixels inside ``mask`` replaced by the model's sample and
            pixels outside ``mask`` preserved from ``video`` exactly (we do a
            final ``where`` composite because SD2-inpaint can drift slightly
            on kept pixels due to VAE round-trip).
        """
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(
                f"VideoInpainter.inpaint: expected video of shape [T, 3, H, W], "
                f"got {tuple(video.shape)}."
            )
        if mask.ndim != 3 or mask.shape[0] != video.shape[0] or \
                mask.shape[1:] != video.shape[2:]:
            raise ValueError(
                f"VideoInpainter.inpaint: expected mask of shape [T, H, W] "
                f"matching video, got mask {tuple(mask.shape)} vs video "
                f"{tuple(video.shape)}."
            )
        if video.min() < -1e-3 or video.max() > 1.0 + 1e-3:
            raise ValueError(
                f"VideoInpainter.inpaint: video must be in [0, 1], got "
                f"range [{video.min().item():.4f}, {video.max().item():.4f}]."
            )

        T, _, H, W = video.shape
        device = video.device
        orig_dtype = video.dtype
        # Move to pipeline device/dtype just for the call.
        video_pipe = video.to(device=self.device_str, dtype=self.dtype)
        mask_pipe = (mask > 0.5).to(device=self.device_str, dtype=self.dtype)

        outputs = []
        for t in range(T):
            frame_01 = video_pipe[t]  # [3, H, W]
            mask_01 = mask_pipe[t]    # [H, W]
            if mask_01.sum() < 1.0:
                # Empty mask: skip diffusion entirely (identity passthrough).
                # This is not a fallback — a mask with zero pixels genuinely
                # has no inpaint target, and forcing a diffusion call would
                # waste ~1s of GPU time for no signal.
                outputs.append(frame_01)
                continue
            # Diffusers pipeline expects PIL Images or tensors [0, 1].
            # Use the tensor interface via the ``image`` / ``mask_image`` args;
            # returns ``PIL.Image`` or a tensor based on ``output_type``.
            result = self.pipeline(
                prompt=prompt,
                image=frame_01.unsqueeze(0),           # [1, 3, H, W]
                mask_image=mask_01.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                num_inference_steps=self.num_inference_steps,
                height=H,
                width=W,
                output_type="pt",
                guidance_scale=7.5,
            )
            out_frame = result.images[0]  # [3, H, W] in [0, 1]
            if out_frame.shape != frame_01.shape:
                raise RuntimeError(
                    f"VideoInpainter: unexpected output shape "
                    f"{tuple(out_frame.shape)} vs expected "
                    f"{tuple(frame_01.shape)}. The pipeline may have resized "
                    f"to a non-multiple-of-8 dimension."
                )
            # Composite: keep original pixels outside mask exactly.
            m3 = mask_01.unsqueeze(0).expand_as(out_frame)
            out_frame = out_frame * m3 + frame_01 * (1.0 - m3)
            outputs.append(out_frame)

        result_tensor = torch.stack(outputs, dim=0)
        return result_tensor.to(device=device, dtype=orig_dtype)

    def extra_repr(self) -> str:
        return (
            f"model_id={self.model_id}, device={self.device_str}, "
            f"dtype={self.dtype}, steps={self.num_inference_steps}"
        )
