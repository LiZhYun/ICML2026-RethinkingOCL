"""TPAMI qualitative-render orchestrator.

Two stages, selectable via --stage:

  picks  : Stage A. For each (arch, dataset), eval per-video ARI on the
           frozen + st (rescue) checkpoints over the first N_CANDIDATES
           validation videos, pick the top-3 scenes by
           delta = ARI(st) - ARI(frozen), write
           tpami-bundle/data/scene_selection_log.json.

  render : Stage B. For each (arch, dataset), load every available
           method checkpoint, run inference on the 3 picked scenes,
           render overlay frames (frame + NEAREST-upscaled hard slot
           mask, stable palette), then emit:
             - tpami-bundle/figures/qualitative/<arch>_<ds>.pdf
               (rows = method, cols = 4 sampled frames of scene 0)
             - tpami-bundle/videos/composite/<arch>_<ds>_scene<k>.mp4
               (all methods side-by-side, ~50 frames @ 8 fps, H.264)

Resumable: skips cells whose output already exists. Reads
tpami-bundle/data/ckpt_manifest.json.

Run on a GPU node (env activated). CPU-only is supported but slow.
"""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.render_utils import (  # noqa: E402
    build_model_and_load,
    get_validation_videos,
    PALETTE,
)

BUNDLE = REPO / "tpami-bundle"
DATA_DIR = "/scratch/work/liz23/slotcontrast/data"
MANIFEST = BUNDLE / "data" / "ckpt_manifest.json"
SCENE_LOG = BUNDLE / "data" / "scene_selection_log.json"
FIG_OUT = BUNDLE / "figures" / "qualitative"
VID_OUT = BUNDLE / "videos" / "composite"
FRAME_TMP = BUNDLE / "videos" / "_frames"
for d in (FIG_OUT, VID_OUT, FRAME_TMP):
    d.mkdir(parents=True, exist_ok=True)

DATASETS = ["movic", "movid", "movie", "ytvis"]
ARCHS = ["sc", "gcv1_pf"]
# Publication display names (broad-reader, academic — no internal codes,
# no seeds). Seed provenance lives in data/ckpt_manifest.json.
ARCH_NAME = {"sc": "SlotContrast", "gcv1_pf": "Grounded Correspondence"}
DS_NAME = {"movic": "MOVi-C", "movid": "MOVi-D",
           "movie": "MOVi-E", "ytvis": "YouTube-VIS 2021"}
# Render order (display label -> manifest method key). Methods missing
# in the manifest for a given cell are silently skipped.
METHOD_ORDER = [
    ("Frozen backbone", "frozen"),
    ("Straight-Through SoftIdent", "st"),
    ("Identity propagation", "nomatch"),
    ("Hungarian matching", "hung"),
    ("Soft SoftIdent", "noST"),
    ("Full fine-tuning", "fullft_lr1e-5"),
]
N_CANDIDATES = 12   # validation videos scanned in Stage A
N_SCENES = 3        # scenes rendered per (arch, dataset)
N_FRAMES = 50       # frames per video in the composite MP4
FPS = 8
MONTAGE_COLS = 4    # frame columns in the static montage


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _attach_step(model, step=100000):
    from types import SimpleNamespace
    model._trainer = SimpleNamespace(
        global_step=int(step), current_epoch=0, is_global_zero=True,
        world_size=1, local_rank=0, global_rank=0,
        datamodule=None, logger=None, log_dir=None, loggers=[])


def load_cell(entry, device):
    cfg = OmegaConf.load(entry["settings"])
    model = build_model_and_load(cfg, Path(entry["ckpt"]), device=device, step=100000)
    _attach_step(model, 100000)
    return cfg, model


@torch.no_grad()
def forward_full(model, batch, device):
    """Run forward + aux_forward; return (outputs, aux_outputs, merged)."""
    bd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    outputs = model(bd)
    aux = model.aux_forward(bd, outputs)
    merged = {**bd, **outputs, **aux}
    return outputs, aux, merged


def soft_masks_from_outputs(outputs, B, T):
    """Extract soft slot masks [B, T, N, h, w] from a forward output."""
    masks = None
    if "decoder" in outputs and isinstance(outputs["decoder"], dict):
        masks = outputs["decoder"].get("masks")
        if masks is None:
            masks = outputs["decoder"].get("masks_resized")
    if masks is None:
        for k in ("decoder_masks", "masks", "predicted_masks", "slot_masks"):
            if k in outputs:
                masks = outputs[k]
                break
    if masks is None:
        raise RuntimeError(f"no masks in outputs keys={list(outputs.keys())}")
    if masks.dim() == 5:
        pass
    elif masks.dim() == 4:
        if masks.shape[0] == B and masks.shape[1] == T:
            HW = masks.shape[-1]
            H = int(round(HW ** 0.5))
            masks = masks.reshape(B, T, masks.shape[2], H, H)
        else:
            masks = masks.view(B, T, *masks.shape[1:])
    elif masks.dim() == 3:
        HW = masks.shape[-1]
        H = int(round(HW ** 0.5))
        masks = masks.reshape(B, T, masks.shape[1], H, H)
    else:
        raise ValueError(f"bad masks shape {tuple(masks.shape)}")
    return masks.float().cpu()


def overlay_frame(frame_chw, mask_nhw, alpha=0.55):
    """frame_chw float/uint8 [3,H,W]; mask_nhw soft [N,h,w].
    Return uint8 [H,W,3] = frame with NEAREST-upscaled argmax slot mask
    alpha-blended on top."""
    f = np.asarray(frame_chw)
    if f.dtype != np.uint8:
        lo, hi = float(f.min()), float(f.max())
        f = (((f - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(f)).clip(0, 255).astype(np.uint8)
    f_hwc = np.transpose(f, (1, 2, 0))  # H,W,3
    H, W = f_hwc.shape[:2]
    slot_id = np.asarray(mask_nhw).argmax(axis=0)  # h,w
    rgb = PALETTE[slot_id % len(PALETTE)].astype(np.uint8)  # h,w,3
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.NEAREST))
    return (((1 - alpha) * f_hwc + alpha * rgb).clip(0, 255)).astype(np.uint8)


# --------------------------------------------------------------------------
# Stage A — scene picking by ARI-improvement proxy
# --------------------------------------------------------------------------
def per_video_ari(model, cfg, device, n_videos):
    """Return list of per-video ARI for the first n_videos val videos."""
    from slotcontrast import metrics
    aris = []
    for idx in range(n_videos):
        batch = get_validation_videos(cfg, data_dir=DATA_DIR, indices=[idx])
        _, _, merged = forward_full(model, batch, device)
        m = metrics.VideoARI(
            ignore_background=True,
            pred_key="decoder_masks_hard",
            true_key="segmentations",
        ).to(device)
        try:
            m.update(**merged)
            aris.append(float(m.compute().item()))
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] video {idx}: ARI failed ({type(e).__name__}: {e}); ARI=nan")
            aris.append(float("nan"))
        m.reset()
    return aris


def stage_picks(manifest, device):
    """Pick scenes ONCE PER DATASET, shared across both architectures, so the
    SC and GCv1 figures/videos show the identical input video (valid
    cross-architecture comparison). Criterion: top-N_SCENES by the MEAN of
    (ARI(st) - ARI(frozen)) over SC and GCv1, on the first N_CANDIDATES
    validation videos. Per-arch raw ARIs are logged for transparency."""
    log = {}
    if SCENE_LOG.exists():
        log = json.loads(SCENE_LOG.read_text())
    for ds in DATASETS:
        if ds in log and log[ds].get("scenes"):
            print(f"[picks] {ds}: already done -> {log[ds]['scenes']}")
            continue
        per_arch = {}
        ok = True
        for arch in ARCHS:
            fz = manifest.get(f"{arch}/{ds}/frozen")
            st = manifest.get(f"{arch}/{ds}/st")
            if not fz or not st:
                print(f"[picks] {ds}/{arch}: missing frozen/st, SKIP dataset")
                ok = False
                break
            print(f"[picks] {ds}/{arch}: scoring {N_CANDIDATES} videos (frozen vs st)")
            _, m_fz = load_cell(fz, device)
            a_fz = per_video_ari(m_fz, OmegaConf.load(fz["settings"]), device, N_CANDIDATES)
            del m_fz
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            _, m_st = load_cell(st, device)
            a_st = per_video_ari(m_st, OmegaConf.load(st["settings"]), device, N_CANDIDATES)
            del m_st
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            dl = [(a_st[i] - a_fz[i])
                  if not (np.isnan(a_st[i]) or np.isnan(a_fz[i])) else np.nan
                  for i in range(N_CANDIDATES)]
            per_arch[arch] = {"frozen_ari": a_fz, "st_ari": a_st, "delta": dl,
                              "frozen_seed": fz["seed"], "st_seed": st["seed"]}
        if not ok:
            continue
        mean_delta = []
        for i in range(N_CANDIDATES):
            vals = [per_arch[a]["delta"][i] for a in ARCHS]
            mean_delta.append(-1e9 if any(np.isnan(v) for v in vals)
                              else float(np.mean(vals)))
        order = sorted(range(N_CANDIDATES), key=lambda i: mean_delta[i], reverse=True)
        picks = order[:N_SCENES]
        log[ds] = {
            "scenes": picks,
            "criterion": ("top-%d by MEAN over {SC,GCv1} of "
                          "delta=ARI(st)-ARI(frozen) over first %d val "
                          "videos; scenes shared across both architectures "
                          "for valid cross-arch comparison"
                          % (N_SCENES, N_CANDIDATES)),
            "mean_delta": mean_delta,
            "per_arch": per_arch,
        }
        SCENE_LOG.write_text(json.dumps(log, indent=2))
        print(f"[picks] {ds}: picks={picks} "
              f"mean_delta={[round(mean_delta[i], 3) for i in picks]}")
    print(f"[picks] DONE -> {SCENE_LOG}")


# --------------------------------------------------------------------------
# Stage B — render montage PDFs + composite MP4s
# --------------------------------------------------------------------------
def render_cell_video(model, cfg, device, scene_idx, n_frames):
    """Return overlay frames [T,H,W,3] uint8 for one video index."""
    batch = get_validation_videos(cfg, data_dir=DATA_DIR, indices=[scene_idx])
    video = batch["video"]                      # [1,T,3,H,W]
    B, T = video.shape[0], video.shape[1]
    outputs, _, _ = forward_full(model, batch, device)
    soft = soft_masks_from_outputs(outputs, B, T)[0]   # [T,N,h,w]
    vid = video[0].cpu().numpy()                        # [T,3,H,W]
    T_use = min(n_frames, vid.shape[0], soft.shape[0])
    return np.stack([overlay_frame(vid[t], soft[t].numpy()) for t in range(T_use)], 0)


def stage_render(manifest, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not SCENE_LOG.exists():
        print("[render] no scene_selection_log.json — run --stage picks first")
        return
    scenes = json.loads(SCENE_LOG.read_text())

    for ds in DATASETS:
        for arch in ARCHS:
            key = f"{arch}/{ds}"
            # scenes are now keyed PER DATASET and shared across archs
            if ds not in scenes or not scenes[ds].get("scenes"):
                print(f"[render] {key}: no scenes for dataset {ds}, SKIP")
                continue
            picks = scenes[ds]["scenes"][:N_SCENES]
            pdf_path = FIG_OUT / f"{arch}_{ds}.pdf"
            mp4_done = all((VID_OUT / f"{arch}_{ds}_scene{k}.mp4").exists()
                           for k in range(len(picks)))
            if pdf_path.exists() and mp4_done:
                print(f"[render] {key}: already rendered, SKIP")
                continue

            # method -> {scene_idx -> frames[T,H,W,3]}
            rendered = {}
            seeds = {}
            for label, mkey in METHOD_ORDER:
                entry = manifest.get(f"{arch}/{ds}/{mkey}")
                if not entry:
                    print(f"[render] {key}: method {mkey} missing, skip row")
                    continue
                print(f"[render] {key}: loading {mkey} (s{entry['seed']})")
                cfg, model = load_cell(entry, device)
                seeds[label] = entry["seed"]
                rendered[label] = {}
                for k, sidx in enumerate(picks):
                    try:
                        rendered[label][k] = render_cell_video(model, cfg, device, sidx, N_FRAMES)
                    except Exception as e:  # noqa: BLE001
                        print(f"    [warn] {mkey} scene{k}(v{sidx}) failed: {e}")
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            labels = [lab for lab, _ in METHOD_ORDER if lab in rendered]

            # ---- static montage PDF: rows=method, cols=4 frames of scene 0
            if not pdf_path.exists() and labels and 0 in next(iter(rendered.values())):
                ncols = MONTAGE_COLS
                fig, axes = plt.subplots(
                    len(labels), ncols,
                    figsize=(2.2 * ncols, 2.2 * len(labels)), squeeze=False)
                for r, lab in enumerate(labels):
                    fr = rendered[lab].get(0)
                    if fr is None:
                        for c in range(ncols):
                            axes[r][c].axis("off")
                        continue
                    Tn = fr.shape[0]
                    cidx = np.linspace(0, Tn - 1, ncols).round().astype(int)
                    for c, t in enumerate(cidx):
                        axes[r][c].imshow(fr[t])
                        axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
                        if c == 0:
                            axes[r][c].set_ylabel(
                                lab, fontsize=10, rotation=90,
                                ha="center", va="center", labelpad=10)
                        if r == 0:
                            axes[r][c].set_title(f"Frame {cidx[c]}", fontsize=10)
                fig.suptitle(
                    f"{ARCH_NAME.get(arch, arch)} — {DS_NAME.get(ds, ds)}",
                    fontsize=13)
                fig.tight_layout()
                fig.savefig(pdf_path, bbox_inches="tight")
                fig.savefig(pdf_path.with_suffix(".png"), dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"[render] wrote {pdf_path}")

            # ---- composite MP4 per scene: methods side-by-side
            for k, sidx in enumerate(picks):
                out_mp4 = VID_OUT / f"{arch}_{ds}_scene{k}.mp4"
                if out_mp4.exists():
                    continue
                panels = [(lab, rendered[lab][k]) for lab in labels
                          if k in rendered.get(lab, {})]
                if not panels:
                    continue
                Tmin = min(p[1].shape[0] for p in panels)
                Hp = min(p[1].shape[1] for p in panels)
                tmpd = FRAME_TMP / f"{arch}_{ds}_scene{k}"
                tmpd.mkdir(parents=True, exist_ok=True)
                for t in range(Tmin):
                    strip = []
                    for lab, fr in panels:
                        im = fr[t]
                        if im.shape[0] != Hp:
                            im = np.array(Image.fromarray(im).resize(
                                (int(im.shape[1] * Hp / im.shape[0]), Hp), Image.NEAREST))
                        # 2px white separator between method panels
                        strip.append(im)
                        strip.append(np.full((Hp, 2, 3), 255, np.uint8))
                    Image.fromarray(np.concatenate(strip[:-1], axis=1)).save(
                        tmpd / f"f{t:04d}.png")
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(tmpd / "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"    [warn] ffmpeg failed for {out_mp4.name}: {r.stderr[:300]}")
                else:
                    print(f"[render] wrote {out_mp4} "
                          f"({len(panels)} panels, {Tmin} frames)")
    print("[render] DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["picks", "render", "all"], required=True)
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tpami-render] stage={args.stage} device={device}")
    if args.stage in ("picks", "all"):
        stage_picks(manifest, device)
    if args.stage in ("render", "all"):
        stage_render(manifest, device)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[tpami-render] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
