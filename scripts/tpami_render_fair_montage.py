"""Re-render the main qualitative artifacts as the FAIR version (montages + videos).

The original montages/videos (tpami_qualitative_render.py) had 6 method panels where
only "Frozen backbone" is un-adapted; the rest are backbone-ADAPTED but the labels did
not say so — so "Identity propagation" read as an isolated mechanism when it is really
*adapted backbone + no predictor* (two variables vs frozen), and there was no frozen
single-variable control.

This regenerates each artifact with:
  (1) a NEW same-seed control row "Frozen + identity propagation" (frozen backbone,
      skip_predictor=true) right under "Frozen backbone";
  (2) every adapted row relabelled to disclose adaptation (publication-canonical method
      names from STYLE_GUIDE §2 + a "Frozen" / "LoRA-adapted" prefix — no internal jargon);
  (3) a dashed group separator between the frozen group and the adapted group.

Static montage : 7 rows × 4 sampled frame-columns  → figures/qualitative/<arch>_<ds>.{png,pdf,svg}
Composite video: the SAME 7 labelled rows animated over time (one evolving column),
                 per picked scene                  → videos/composite/<arch>_<ds>_scene<k>.mp4

Reuses the validated render path from tpami_qualitative_render (load_cell,
render_cell_video) — does NOT modify it. Reads the frozen_nomatch entries in
data/ckpt_manifest.json. Same picked scenes as the originals.

  python3 scripts/tpami_render_fair_montage.py                       # montages, all 8
  python3 scripts/tpami_render_fair_montage.py --videos              # montages + videos, all 8
  python3 scripts/tpami_render_fair_montage.py --videos --cells gcv1_pf/movic
Run on a GPU node (inference only).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.tpami_qualitative_render import (  # noqa: E402
    load_cell, render_cell_video, ARCH_NAME, DS_NAME, MONTAGE_COLS,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

BUNDLE = REPO / "tpami-bundle"
MANIFEST = json.loads((BUNDLE / "data" / "ckpt_manifest.json").read_text())
SCENES = json.loads((BUNDLE / "data" / "scene_selection_log.json").read_text())
FIG_OUT = BUNDLE / "figures" / "qualitative"
VID_OUT = BUNDLE / "videos" / "composite"
FRAME_TMP = BUNDLE / "videos" / "_frames_fair"
N_FRAMES = 50   # full clip; montage samples MONTAGE_COLS, video plays all
N_SCENES = 3    # picked scenes rendered per cell as videos (matches original)
FPS = 8

ARCHS = ["sc", "gcv1_pf"]
DSS = ["movic", "movid", "movie", "ytvis"]

# (display label, manifest key, group). group: "frozen" | "adapted"
# Publication-canonical names (STYLE_GUIDE §2); adaptation disclosed via the
# "Frozen" / "LoRA-adapted" prefix + the group separator. No internal jargon.
ROWS = [
    ("Frozen backbone",                            "frozen",         "frozen"),
    ("Frozen +\nidentity propagation",             "frozen_nomatch", "frozen"),
    ("LoRA-adapted +\nstraight-through correspondence", "st",        "adapted"),
    ("LoRA-adapted +\nidentity propagation",       "nomatch",        "adapted"),
    ("LoRA-adapted +\nHungarian matching",         "hung",           "adapted"),
    ("LoRA-adapted +\nsoft correspondence",        "noST",           "adapted"),
    ("Full fine-tuning",                           "fullft_lr1e-5",  "adapted"),
]


def _add_group_separator(fig, axes_col0, groups, captions=True, x0=0.04):
    """Dashed line (+ optional captions) between the last 'frozen' row and first 'adapted' row."""
    if "frozen" not in groups or "adapted" not in groups:
        return
    last_frozen = max(i for i, g in enumerate(groups) if g == "frozen")
    first_adapt = min(i for i, g in enumerate(groups) if g == "adapted")
    ysep = (axes_col0[last_frozen].get_position().y0
            + axes_col0[first_adapt].get_position().y1) / 2.0
    fig.add_artist(mlines.Line2D([x0, 0.99], [ysep, ysep], color="0.4", lw=1.1,
                                 ls="--", transform=fig.transFigure, clip_on=False))
    if captions:
        fig.text(0.985, ysep + 0.003, "frozen backbone (no adaptation)", fontsize=8,
                 style="italic", ha="right", va="bottom", color="0.35")
        fig.text(0.985, ysep - 0.003, "backbone adapted (LoRA / full fine-tuning)",
                 fontsize=8, style="italic", ha="right", va="top", color="0.35")


def load_rows(arch, ds, device, scene_ks):
    """Load each method once; render the needed scenes' full clips.
    Returns list of (label, group, {scene_k: frames[T,H,W,3]})."""
    rows = []
    for label, mkey, group in ROWS:
        entry = MANIFEST.get(f"{arch}/{ds}/{mkey}")
        if not entry:
            print(f"[render] {arch}/{ds}: {mkey} missing, skip row")
            continue
        cfg, m = load_cell(entry, device)
        per_scene = {}
        for k in scene_ks:
            sidx = SCENES[ds]["scenes"][k]
            per_scene[k] = render_cell_video(m, cfg, device, sidx, N_FRAMES)
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        rows.append((label, group, per_scene))
    return rows


def build_montage(arch, ds, rows):
    """7-row × MONTAGE_COLS static montage from scene-0 clips."""
    ncol = MONTAGE_COLS
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), squeeze=False)
    Tref = rows[0][2][0].shape[0]
    cidx = np.linspace(0, Tref - 1, ncol).round().astype(int)
    for r, (lab, group, per_scene) in enumerate(rows):
        fr = per_scene[0]
        ci = np.linspace(0, fr.shape[0] - 1, ncol).round().astype(int)
        for c in range(ncol):
            ax = axes[r][c]
            ax.imshow(fr[ci[c]]); ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=8.5, rotation=90, ha="center",
                              va="center", labelpad=10)
            if r == 0:
                ax.set_title(f"Frame {cidx[c]}", fontsize=10)
    fig.suptitle(f"{ARCH_NAME.get(arch, arch)} — {DS_NAME.get(ds, ds)}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _add_group_separator(fig, [axes[r][0] for r in range(nrow)],
                         [g for _, g, _ in rows])
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    base = FIG_OUT / f"{arch}_{ds}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] montage figures/qualitative/{arch}_{ds}.{{png,pdf,svg}}  ({nrow} rows)")


def build_video(arch, ds, rows, k):
    """Animated montage: same 7 labelled rows, one evolving column, scene k → mp4."""
    ncol = len(rows)
    Tmin = min(per[k].shape[0] for _, _, per in rows)
    # HORIZONTAL layout: the 7 methods side-by-side (1 row x N cols), animated;
    # academic labels as wrapped column titles, vertical dashed group separator.
    fig, axes = plt.subplots(1, ncol, figsize=(2.0 * ncol, 3.0), squeeze=False)
    ims = []
    for c, (lab, group, per) in enumerate(rows):
        ax = axes[0][c]
        im = ax.imshow(per[k][0]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab, fontsize=7)
        ims.append(im)
    fig.suptitle(f"{ARCH_NAME.get(arch, arch)} — {DS_NAME.get(ds, ds)}", fontsize=12, y=0.99)
    # FIXED layout (constant frame size for ffmpeg); reserve top band for titles.
    fig.subplots_adjust(left=0.004, right=0.996, top=0.70, bottom=0.01, wspace=0.05)
    groups = [g for _, g, _ in rows]
    if "frozen" in groups and "adapted" in groups:
        last_frozen = max(i for i, g in enumerate(groups) if g == "frozen")
        first_adapt = min(i for i, g in enumerate(groups) if g == "adapted")
        xsep = (axes[0][last_frozen].get_position().x1
                + axes[0][first_adapt].get_position().x0) / 2.0
        fig.add_artist(mlines.Line2D([xsep, xsep], [0.01, 0.78], color="0.4", lw=1.1,
                                     ls="--", transform=fig.transFigure, clip_on=False))

    tmpd = FRAME_TMP / f"{arch}_{ds}_scene{k}"
    tmpd.mkdir(parents=True, exist_ok=True)
    for t in range(Tmin):
        for r, (_, _, per) in enumerate(rows):
            ims[r].set_data(per[k][t])
        fig.savefig(tmpd / f"f{t:04d}.png", dpi=110)
    plt.close(fig)

    VID_OUT.mkdir(parents=True, exist_ok=True)
    out_mp4 = VID_OUT / f"{arch}_{ds}_scene{k}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", str(tmpd / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for f in tmpd.glob("f*.png"):
        f.unlink()
    tmpd.rmdir()
    if r.returncode != 0:
        print(f"[ERR] ffmpeg {out_mp4.name}: {r.stderr[:200]}")
    else:
        print(f"[ok] video videos/composite/{arch}_{ds}_scene{k}.mp4  ({ncol} panels, {Tmin} frames)")


def render_cell(arch, ds, device, do_videos):
    n_scn = min(N_SCENES, len(SCENES[ds]["scenes"]))
    scene_ks = list(range(n_scn)) if do_videos else [0]
    rows = load_rows(arch, ds, device, scene_ks)
    if not rows:
        print(f"[skip] {arch}/{ds}: no rows")
        return
    build_montage(arch, ds, rows)
    if do_videos:
        for k in scene_ks:
            build_video(arch, ds, rows, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None,
                    help="subset like gcv1_pf/movic sc/movid; default all 8")
    ap.add_argument("--videos", action="store_true", help="also (re)render composite videos")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = ([tuple(c.split("/")) for c in args.cells] if args.cells
             else [(a, d) for d in DSS for a in ARCHS])
    for arch, ds in pairs:
        try:
            render_cell(arch, ds, device, args.videos)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[ERR] {arch}/{ds}: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
