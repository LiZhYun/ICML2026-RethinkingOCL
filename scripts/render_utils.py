"""M3: Per-slot tracking visualization.

For each Phase 5 instrumented-run checkpoint at the final step (e.g. 30K),
runs inference on a small set of fixed validation videos and saves the
predicted slot masks per frame as PNG overlays. Slots are color-coded
*consistently across frames* using a fixed palette indexed by slot id.

The qualitative claim:
  • ST: slot identity is stable across frames (object permanence preserved
    even through occlusions / fast motion).
  • Hungarian: identity *can* flip at fast motion (the matching gradient is
    zero, so the backbone doesn't learn matchable representations and
    Hungarian must work harder per-frame).
  • nomatch: index slots drift — same object ends up under different
    indices on different frames.
  • frozen baseline: depends on the backbone's prior; usually noisy.

This script intentionally produces *raw* per-frame PNGs + a side-by-side
PDF montage. It does NOT auto-judge tracking quality — the user inspects
the visual evidence.

Usage:
  python scripts/m3_slot_tracking_viz.py \
      --logs-root logs/icml27_oral \
      --variants instr_movic_gcv1_st instr_movic_gcv1_hung \
                 instr_movic_gcv1_noST instr_movic_gcv1_nomatch \
                 instr_movic_gcv1_frozen \
      --video-indices 0 3 7 12

Outputs:
  review-stage/analysis/slot_viz/<variant>/video_<idx>/frame_<t>.png
  review-stage/analysis/slot_viz/M3_slot_tracking_montage.pdf
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "review-stage/analysis/slot_viz"
OUT.mkdir(parents=True, exist_ok=True)


# Stable color palette (matplotlib tab20). 20 distinct colors — enough for
# any standard slot count. Indexed by slot id, consistent across frames.
PALETTE = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
    [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
    [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
], dtype=np.uint8)


def find_run_dirs(logs_root: Path, name_glob: str) -> List[Path]:
    return sorted(d for d in logs_root.glob(f"*{name_glob}*")
                  if d.is_dir() and (d / "checkpoints").is_dir())


def find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    ckpts = list((run_dir / "checkpoints").glob("step=*.ckpt"))
    if not ckpts:
        return None
    def step_of(p):
        m = re.match(r"step=(\d+)", p.name)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=step_of)


def load_settings(run_dir: Path):
    sp = run_dir / "settings.yaml"
    if not sp.exists():
        raise FileNotFoundError(f"settings.yaml not found at {sp}")
    return OmegaConf.load(sp)


def _attach_fake_trainer(model, step: int = 30000):
    from types import SimpleNamespace
    model._trainer = SimpleNamespace(
        global_step=int(step), current_epoch=0, is_global_zero=True,
        world_size=1, local_rank=0, global_rank=0,
        datamodule=None, logger=None, log_dir=None, loggers=[])


def build_model_and_load(cfg, ckpt_path: Path, device: str, step: int = 30000):
    from slotcontrast import models, metrics
    train_metrics = ({n: metrics.build(c) for n, c in cfg.train_metrics.items()}
                     if cfg.get("train_metrics") else None)
    val_metrics = ({n: metrics.build(c) for n, c in cfg.val_metrics.items()}
                   if cfg.get("val_metrics") else None)
    model = models.build(cfg.model, cfg.optimizer, train_metrics, val_metrics)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)
    model.load_state_dict(state, strict=False)
    model.eval()
    _attach_fake_trainer(model, step=step)
    return model.to(device)


def get_validation_videos(cfg, data_dir: str, indices: List[int]) -> dict:
    """Materialize the requested validation video indices into one batch.

    We iterate the val loader, stockpiling videos until we've collected the
    requested indices (interpreted as the order they appear in the loader,
    not absolute dataset indices). Returns a single batch with shape
    matching the model's expected input.
    """
    from slotcontrast import data as data_module
    dataset = data_module.build(cfg.dataset, data_dir=data_dir)
    dataset.setup("fit")
    val_loader = dataset.val_dataloader() or dataset.train_dataloader()
    max_idx = max(indices) + 1
    collected = {}
    cursor = 0
    for batch in val_loader:
        bsz = batch["video"].shape[0] if "video" in batch else next(iter(batch.values())).shape[0]
        for i in range(bsz):
            if cursor in indices:
                for k, v in batch.items():
                    collected.setdefault(k, []).append(v[i:i+1] if torch.is_tensor(v) else v)
            cursor += 1
            if cursor >= max_idx:
                break
        if cursor >= max_idx:
            break
    out = {}
    for k, lst in collected.items():
        if torch.is_tensor(lst[0]):
            # YT-VIS validation videos have variable T per video; truncate to the
            # min frame count across the requested videos so torch.cat works.
            if lst[0].dim() >= 2 and any(t.shape[1] != lst[0].shape[1] for t in lst):
                t_min = min(t.shape[1] for t in lst)
                lst = [t[:, :t_min] for t in lst]
            out[k] = torch.cat(lst, dim=0)
        else:
            out[k] = lst
    return out


@torch.no_grad()
def predict_masks(model, batch: dict, device: str):
    """Run forward and return predicted slot masks [B, T, N, H, W] in [0, 1]."""
    batch_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch_d)
    # Common locations for predicted masks across slotcontrast variants
    masks = None
    if "decoder" in out and isinstance(out["decoder"], dict):
        m1 = out["decoder"].get("masks")
        m2 = out["decoder"].get("masks_resized")
        masks = m1 if m1 is not None else m2
    if masks is None:
        for key in ("masks", "predicted_masks", "slot_masks"):
            if key in out:
                masks = out[key]
                break
    if masks is None:
        raise RuntimeError(f"could not find predicted masks in model output keys: {list(out.keys())}")
    # Expected shape: [B, T, N, H, W] or [B, T, N, HW] or [B*T, N, H, W].
    # The decoder typically emits [B, T, N, HW]; older code paths emit
    # [B*T, N, H, W]. Handle both robustly.
    B_in = batch_d["video"].shape[0]
    T_in = batch_d["video"].shape[1]
    if masks.dim() == 5:
        pass  # already [B, T, N, H, W]
    elif masks.dim() == 4:
        if masks.shape[0] == B_in and masks.shape[1] == T_in:
            # [B, T, N, HW] — square-root the last dim
            HW = masks.shape[-1]
            H = int(round(HW ** 0.5))
            assert H * H == HW, f"non-square mask flattening: HW={HW}"
            masks = masks.reshape(B_in, T_in, masks.shape[2], H, H)
        else:
            # [B*T, N, H, W] -> [B, T, N, H, W]
            B = masks.shape[0] // T_in
            masks = masks.view(B, T_in, *masks.shape[1:])
    elif masks.dim() == 3:
        # [B*T, N, HW]
        HW = masks.shape[-1]
        H = int(round(HW ** 0.5))
        assert H * H == HW
        B = masks.shape[0] // T_in
        masks = masks.reshape(B, T_in, masks.shape[1], H, H)
    else:
        raise ValueError(f"unexpected masks shape {tuple(masks.shape)}")
    return masks.cpu()


def colorize(masks_t: np.ndarray) -> np.ndarray:
    """masks_t: [N, H, W] in [0,1] (per-slot soft mask).
    Argmax over slots, then map to PALETTE."""
    N, H, W = masks_t.shape
    slot_id = masks_t.argmax(axis=0)  # [H, W]
    rgb = PALETTE[slot_id % len(PALETTE)]  # [H, W, 3]
    return rgb


def overlay(frame_rgb: np.ndarray, slot_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend slot coloring on top of the original frame."""
    return (alpha * slot_rgb + (1 - alpha) * frame_rgb).clip(0, 255).astype(np.uint8)


def save_png(path: Path, arr: np.ndarray):
    from PIL import Image
    Image.fromarray(arr).save(path)


def denorm_video(video_t: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] in approx ImageNet-normalized → [T, H, W, 3] uint8 in [0, 255]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    v = (video_t * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return (v * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default="logs/icml27_oral", type=Path)
    ap.add_argument("--data-dir", default="/scratch/work/liz23/slotcontrast/data", type=str)
    ap.add_argument("--variants", nargs="+", default=[
        "instr_movic_gcv1_st",
        "instr_movic_gcv1_hung",
        "instr_movic_gcv1_noST",
        "instr_movic_gcv1_nomatch",
        "instr_movic_gcv1_frozen",
    ])
    ap.add_argument("--video-indices", nargs="+", type=int, default=[0, 3, 7, 12])
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Override output directory (default: review-stage/analysis/slot_viz/)")
    args = ap.parse_args()
    if args.out_dir is not None:
        global OUT
        OUT = Path(args.out_dir)
        OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    montage_data = {}  # variant -> video_idx -> [T, H, W, 3] overlays

    for variant in args.variants:
        run_dirs = find_run_dirs(args.logs_root, variant)
        if not run_dirs:
            print(f"[{variant}] no run dirs found under {args.logs_root}; skipping")
            continue
        run_dir = run_dirs[0]
        ckpt = find_latest_ckpt(run_dir)
        if ckpt is None:
            print(f"[{variant}] no checkpoints yet; skipping")
            continue
        print(f"\n=== {variant} (ckpt={ckpt.name}) ===")
        cfg = load_settings(run_dir)

        try:
            batch = get_validation_videos(cfg, args.data_dir, args.video_indices)
            step_match = re.match(r".*step=(\d+)", str(ckpt))
            step_val = int(step_match.group(1)) if step_match else 30000
            model = build_model_and_load(cfg, ckpt, device, step=step_val)
            masks = predict_masks(model, batch, device)  # [B, T, N, H, W]
            videos = batch["video"]                       # [B, T, C, H, W]
        except Exception as e:
            print(f"  failed: {e}")
            continue

        B, T, N, Hm, Wm = masks.shape
        print(f"  masks shape {tuple(masks.shape)}")
        for bi, vid_idx in enumerate(args.video_indices[:B]):
            v = denorm_video(videos[bi])  # [T, Hv, Wv, 3]
            # If mask resolution differs from video, upscale mask via nearest
            if (Hm, Wm) != v.shape[1:3]:
                mt = torch.nn.functional.interpolate(
                    masks[bi].view(T * N, 1, Hm, Wm),
                    size=v.shape[1:3], mode="nearest"
                ).view(T, N, *v.shape[1:3]).numpy()
            else:
                mt = masks[bi].numpy()
            overlays = []
            vid_dir = OUT / variant / f"video_{vid_idx:03d}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            for t in range(T):
                slot_rgb = colorize(mt[t])
                ov = overlay(v[t], slot_rgb, alpha=0.55)
                save_png(vid_dir / f"frame_{t:03d}.png", ov)
                overlays.append(ov)
            montage_data.setdefault(variant, {})[vid_idx] = np.stack(overlays, axis=0)

            # Stitch PNG sequence into MP4 via ffmpeg if available.
            mp4_path = vid_dir / "tracking.mp4"
            try:
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", "6",
                    "-i", str(vid_dir / "frame_%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    str(mp4_path),
                ], check=True)
                print(f"  wrote {mp4_path}")
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"  mp4 stitch skipped ({type(e).__name__})")

    if not montage_data:
        print("No variants produced output; skipping montage.")
        return

    # Build a montage PDF: rows = variants, columns = (video, frame)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Show 4 evenly-spaced frames per video for compactness
        n_frames_show = 4
        videos_in_order = sorted({vi for vd in montage_data.values() for vi in vd})
        n_vids = len(videos_in_order)
        n_rows = len(montage_data)
        n_cols = n_vids * n_frames_show
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(1.5 * n_cols, 1.5 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
        for r, (variant, by_vid) in enumerate(montage_data.items()):
            for ci, vi in enumerate(videos_in_order):
                if vi not in by_vid:
                    for k in range(n_frames_show):
                        axes[r, ci * n_frames_show + k].axis("off")
                    continue
                clip = by_vid[vi]  # [T, H, W, 3]
                T = clip.shape[0]
                idxs = np.linspace(0, T - 1, n_frames_show).astype(int)
                for k, t in enumerate(idxs):
                    ax = axes[r, ci * n_frames_show + k]
                    ax.imshow(clip[t])
                    ax.set_xticks([]); ax.set_yticks([])
                    if k == 0 and ci == 0:
                        ax.set_ylabel(variant.replace("instr_movic_gcv1_", ""), fontsize=9)
                    if r == 0 and k == 0:
                        ax.set_title(f"vid {vi}", fontsize=9)
        fig.suptitle("Per-slot tracking — color = slot id (consistent across frames)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_pdf = OUT / "M3_slot_tracking_montage.pdf"
        fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
        print(f"\nWrote {out_pdf}")
    except Exception as e:
        print(f"Montage failed: {e}")

    # Side-by-side variant comparison videos: for each video index, stack
    # all variants vertically into a single MP4 so reviewers can scrub one
    # video and watch all 5 predictors track in lockstep.
    try:
        videos_in_order = sorted({vi for vd in montage_data.values() for vi in vd})
        for vi in videos_in_order:
            stacks = []
            labels = []
            for variant, by_vid in montage_data.items():
                if vi in by_vid:
                    stacks.append(by_vid[vi])  # [T, H, W, 3]
                    labels.append(variant.replace("instr_movic_gcv1_", ""))
            if len(stacks) < 2:
                continue
            T_all = min(s.shape[0] for s in stacks)
            H, W, _ = stacks[0].shape[1:]
            comp_dir = OUT / f"_compare_video_{vi:03d}"
            comp_dir.mkdir(parents=True, exist_ok=True)
            # Annotate each row with variant name (PIL).
            from PIL import Image, ImageDraw, ImageFont
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()
            # Pretty variant names for the broader-reader audience.
            pretty = {
                "st": "ST-SoftIdent (ours)",
                "hung": "Hungarian (hard)",
                "noST": "soft Sinkhorn (no-ST)",
                "nomatch": "no temporal matching",
                "frozen": "frozen backbone",
            }
            for t in range(T_all):
                # HORIZONTAL stack (1 x N) with header label above each panel
                # and a frame counter on the bottom.
                HEADER_H = 28
                FOOTER_H = 24
                panels = []
                for label, clip in zip(labels, stacks):
                    nice = pretty.get(label, label)
                    panel_img = clip[t]
                    panel_h, panel_w, _ = panel_img.shape
                    composite = np.zeros(
                        (panel_h + HEADER_H + FOOTER_H, panel_w, 3),
                        dtype=np.uint8)
                    composite[HEADER_H:HEADER_H + panel_h] = panel_img
                    img = Image.fromarray(composite)
                    draw = ImageDraw.Draw(img)
                    # Header band (black) with variant name (white)
                    draw.rectangle((0, 0, panel_w, HEADER_H), fill=(20, 20, 20))
                    try:
                        bbox = draw.textbbox((0, 0), nice, font=font)
                        tw = bbox[2] - bbox[0]
                    except Exception:
                        tw = 8 * len(nice)
                    draw.text(((panel_w - tw) // 2, 6), nice,
                              font=font, fill=(255, 255, 255))
                    # Footer band with frame counter
                    fy = HEADER_H + panel_h
                    draw.rectangle((0, fy, panel_w, fy + FOOTER_H),
                                   fill=(40, 40, 40))
                    counter = f"frame {t+1:02d} / {T_all:02d}"
                    try:
                        bbox = draw.textbbox((0, 0), counter, font=font)
                        cw = bbox[2] - bbox[0]
                    except Exception:
                        cw = 8 * len(counter)
                    draw.text(((panel_w - cw) // 2, fy + 4), counter,
                              font=font, fill=(200, 200, 200))
                    panels.append(np.array(img))
                stacked = np.concatenate(panels, axis=1)  # horizontal stack
                Image.fromarray(stacked).save(comp_dir / f"frame_{t:03d}.png")
            try:
                import subprocess
                mp4_path = OUT / f"M3_compare_video_{vi:03d}.mp4"
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", "6",
                    "-i", str(comp_dir / "frame_%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    str(mp4_path),
                ], check=True)
                print(f"wrote {mp4_path}")
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                print(f"compare mp4 skipped ({type(e).__name__})")
    except Exception as e:
        print(f"Side-by-side compare failed: {e}")


if __name__ == "__main__":
    main()
