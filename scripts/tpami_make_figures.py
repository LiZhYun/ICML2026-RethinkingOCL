"""Regenerate ALL quantitative figures from v7.5 sources with the single
canonical publication naming (STYLE_GUIDE §2). Replaces the stale oral-era
fig_*.{pdf,png} (which had n=3 numbers + a 3rd naming scheme).

No GPU. Sources (same as tables.tex):
  /scratch/elec/t41020_egovla/paired_stats.md        F1/F3 forest
  /scratch/elec/t41020_egovla/v2_grid_summary.md     predictor ablation,
                                                     featrec/rank/unfreeze
                                                     sweeps, D1-light
  EXPERIMENTAL_RESULTS.md §8.2.7                      LoRA-LR robustness
  <cell>/metrics/version_0/metrics.csv               convergence /
                                                     feature-PR / drift
Cell dirs come from tpami-bundle/data/ckpt_manifest.json.
"""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ELEC = Path("/scratch/elec/t41020_egovla")
BUNDLE = REPO / "tpami-bundle"
FIG = BUNDLE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- the ONE canonical naming (matches STYLE_GUIDE §2 + qualitative figs)
DISPLAY = {
    "frozen": "Frozen backbone",
    "st": "Straight-Through SoftIdent",
    "nomatch": "Identity propagation",
    "hung": "Hungarian matching",
    "noST": "Soft SoftIdent",
    "fullft": "Full fine-tuning",
}
ARCH_NAME = {"sc": "SlotContrast", "gcv1_pf": "Grounded Correspondence"}
DS_NAME = {"movic": "MOVi-C", "movid": "MOVi-D",
           "movie": "MOVi-E", "ytvis": "YouTube-VIS 2021"}
C_BASE, C_OURS, C_ALT, C_NS = "#7f7f7f", "#1f77b4", "#ff7f0e", "#bbbbbb"
plt.rcParams.update({"font.size": 11})


def md_rows(text, header_re, ncols):
    m = re.search(header_re, text)
    if not m:
        return []
    rows, started = [], False
    for line in text[m.end():].splitlines():
        s = line.strip()
        if s.startswith("|"):
            started = True
            c = [x.strip() for x in s.strip("|").split("|")]
            if len(c) < ncols or set("".join(c)) <= set("-: "):
                continue
            if c[0].lower() in ("comparison", "variant", "dataset",
                                "method", "sorted by raw p", "cell",
                                "arch", "backbone"):
                continue
            rows.append(c)
        elif started and s == "":
            continue
        elif started:
            break
    return rows


def num(s):
    m = re.search(r"[-+]?\d+\.?\d*", s.replace("−", "-"))
    return float(m.group()) if m else np.nan


# ---------------------------------------------------------------- forest
def fig_rescue_forest():
    ps = (ELEC / "paired_stats.md").read_text()
    fams = [("### F1 — SC rescue lift (st − frozen), 12 tests", "SlotContrast"),
            ("### F3 — GCv1 rescue lift (st − frozen), 12 tests",
             "Grounded Correspondence")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, (hdr, title) in zip(axes, fams):
        rows = md_rows(ps, re.escape(hdr), 7)
        labs, mu, lo, hi, rej = [], [], [], [], []
        for r in rows:
            comp = r[0].split(":")[0].strip()
            ci = re.findall(r"[-+]?\d+\.?\d*", r[3].replace("−", "-"))
            if len(ci) < 2:
                continue
            labs.append(comp.replace("SC ", "").replace("GCv1 ", ""))
            mu.append(num(r[2])); lo.append(float(ci[0])); hi.append(float(ci[1]))
            rej.append("yes" in r[6].lower())
        y = np.arange(len(labs))[::-1]
        for i in range(len(labs)):
            col = C_OURS if rej[i] else C_NS
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=col, lw=2.4, zorder=2)
            ax.plot(mu[i], y[i], "o", color=col, ms=7, zorder=3)
        ax.axvline(0, color="k", lw=1, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Δ vs Frozen backbone  (paired-bootstrap 95% CI, n=5)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Rescue protocol lift over the frozen baseline", fontsize=13)
    _save(fig, "fig_rescue_forest")


# ------------------------------------------------ predictor ablation bars
def _grid_block(grid, arch_hdr, ds):
    gi = grid.find(arch_hdr)
    if gi < 0:
        return {}
    seg = grid[gi + len(arch_hdr):]
    nx = re.search(r"\n## ", seg)
    seg = seg[:nx.start()] if nx else seg
    di = seg.find(f"### {ds}")
    if di < 0:
        return {}
    sub = seg[di + len(f"### {ds}"):]
    nx = re.search(r"\n### ", sub)
    sub = sub[:nx.start()] if nx else sub
    out = {}
    for line in sub.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        c = [x.strip() for x in s.strip("|").split("|")]
        if len(c) < 5 or set("".join(c)) <= set("-: "):
            continue
        k = c[0].lower()
        if "frozen backbone" in k: out["frozen"] = num(c[3])
        elif "rescue headline" in k: out["st"] = num(c[3])
        elif "no predictor matching" in k: out["nomatch"] = num(c[3])
        elif "hungarian predictor" in k: out["hung"] = num(c[3])
        elif "soft softident" in k: out["noST"] = num(c[3])
    return out


def fig_predictor_ablation():
    grid = (ELEC / "v2_grid_summary.md").read_text()
    variants = ["st", "hung", "noST", "nomatch"]
    cols = {"st": C_OURS, "hung": C_ALT, "noST": "#2ca02c", "nomatch": C_BASE}
    dss = [("movic", "MOVi-C"), ("movid", "MOVi-D"),
           ("movie", "MOVi-E"), ("ytvis", "YouTube-VIS 2021")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (arch_hdr, aname) in zip(
            axes, [("## SlotContrast", "SlotContrast"),
                   ("## Grounded Correspondence", "Grounded Correspondence")]):
        x = np.arange(len(dss)); w = 0.2
        for j, v in enumerate(variants):
            deltas = []
            for code, _ in dss:
                b = _grid_block(grid, arch_hdr, _)
                deltas.append((b.get(v, np.nan) - b.get("frozen", np.nan))
                              if "frozen" in b else np.nan)
            ax.bar(x + (j - 1.5) * w, deltas, w, label=DISPLAY[v],
                   color=cols[v])
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([d[1] for d in dss], fontsize=9)
        ax.set_title(aname, fontsize=12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Δ ARI vs Frozen backbone (%, n=5)")
    axes[1].legend(fontsize=9, loc="lower left", framealpha=0.9)
    fig.suptitle("Temporal-predictor ablation under the rescue scaffold",
                 fontsize=13)
    _save(fig, "fig_predictor_ablation")


# ----------------------------------------------------- ablation sweeps
def _sweep(grid, sub_hdr):
    m = re.search(re.escape(sub_hdr), grid)
    seg = grid[m.end():] if m else ""
    out = []
    for line in seg.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            if out:
                break
            continue
        c = [x.strip() for x in s.strip("|").split("|")]
        if len(c) < 4 or set("".join(c)) <= set("-: ") or c[0].lower() == "variant":
            continue
        out.append((c[0], num(c[3])))
    return out


def _ablation(sub_hdr, default_label, default_val, xlabel, fname, title):
    grid = (ELEC / "v2_grid_summary.md").read_text()
    pts = _sweep(grid, sub_hdr)
    pts.append((default_label, default_val))
    order = {"step 0 (immediate)": 0, "step 2500": 2500, "step 5000 (default)": 5000,
             "step 10000": 10000, "r=4": 4, "r=8 (default)": 8, "r=16": 16,
             "r=32": 32, "w=0.5": .5, "w=1.0": 1.0, "w=1.5 (default)": 1.5,
             "w=2.0": 2.0}
    pts = sorted(pts, key=lambda p: order.get(p[0], 1e9))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [p[0].replace(" (default)", "\n(default)") for p in pts]
    ys = [p[1] for p in pts]
    bars = ax.bar(range(len(xs)), ys,
                  color=[C_OURS if "default" in p[0] else C_BASE for p in pts])
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=9)
    ax.set_ylabel("ARI (%, n=5)"); ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12); ax.grid(axis="y", alpha=0.25)
    for b, yv in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, yv + 0.6, f"{yv:.1f}",
                ha="center", fontsize=8)
    _save(fig, fname)


# ---------------------------------------------------------- D1-light
def fig_d1light():
    grid = (ELEC / "v2_grid_summary.md").read_text()
    out = {}
    for code, name in [("movid", "MOVi-D"), ("movie", "MOVi-E")]:
        b = _grid_block(grid, "## SlotContrast", name)  # ensures section parsed
        seg = grid
        # collect the three D1-light rows within SC <ds> block
        gi = seg.find("## SlotContrast")
        s2 = seg[gi:].split("## Grounded")[0]
        di = s2.find(f"### {name}")
        sub = s2[di:].split("### ", 2)
        block = sub[1] if len(sub) > 1 else s2[di:]
        vals = {}
        for line in block.splitlines():
            if "D1-light + default rescue" in line: vals["default"] = num(line.split("|")[4])
            elif "D1-light + ST rescue" in line: vals["st"] = num(line.split("|")[4])
            elif "D1-light + nomatch" in line: vals["nomatch"] = num(line.split("|")[4])
        out[name] = vals
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cats = ["default", "st", "nomatch"]
    catlab = {"default": "Default predictor", "st": "Straight-Through SoftIdent",
              "nomatch": "Identity propagation"}
    x = np.arange(len(cats)); w = 0.35
    for j, (ds, vv) in enumerate(out.items()):
        ax.bar(x + (j - .5) * w, [vv.get(c, np.nan) for c in cats], w, label=ds,
               color=[C_OURS, C_ALT][j])
    ax.set_xticks(x); ax.set_xticklabels([catlab[c] for c in cats], fontsize=9)
    ax.set_ylabel("ARI (%, n=5)"); ax.legend(title="SlotContrast w/o slot-slot loss")
    ax.set_title("D1-light: SlotContrast without the slot-slot contrastive loss",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig_d1light")


# ---------------------------------------- trajectory figs from metrics.csv
def _traj(csv_path, ycol):
    xs, ys = [], []
    with open(csv_path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            v = row.get(ycol, "")
            st = row.get("step", "")
            if v not in ("", None) and st not in ("", None):
                try:
                    ys.append(float(v)); xs.append(float(st))
                except ValueError:
                    pass
    return np.array(xs), np.array(ys)


def _traj_fig(ycol, ylabel, title, fname, scale=1.0):
    man = json.loads((BUNDLE / "data" / "ckpt_manifest.json").read_text())
    cells = [("sc/movid/frozen", "frozen"), ("sc/movid/st", "st"),
             ("sc/movid/nomatch", "nomatch")]
    cols = {"frozen": C_BASE, "st": C_OURS, "nomatch": C_ALT}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    seeds = []
    for key, code in cells:
        e = man.get(key)
        if not e:
            continue
        mp = Path(e["dir"]) / "metrics" / "version_0" / "metrics.csv"
        if not mp.exists():
            continue
        x, y = _traj(mp, ycol)
        if len(x) == 0:
            continue
        ax.plot(x, y * scale, color=cols[code], lw=2, label=DISPLAY[code])
        seeds.append(f"{DISPLAY[code]}: s{e['seed']}")
    ax.set_xlabel("Training step"); ax.set_ylabel(ylabel)
    ax.set_title(title + "  (SlotContrast × MOVi-D, representative run)",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    _save(fig, fname)
    return seeds


# ------------------------------------------------- §8.2.7 LR-robustness
def fig_lora_lr_swing():
    er = (REPO / "EXPERIMENTAL_RESULTS.md").read_text()
    rows = md_rows(
        er,
        r"\| Method \| lr=1×10⁻⁵ \| lr=4×10⁻⁶ \| lr=4×10⁻⁵ \(canon\) \| "
        r"lr=1×10⁻⁴ \| lr=4×10⁻⁴ \| LR-swing \|", 7)
    # column LRs in ascending order: 4e-6,1e-5,4e-5,1e-4,4e-4
    lrs = [4e-6, 1e-5, 4e-5, 1e-4, 4e-4]
    colmap = [2, 1, 3, 4, 5]  # md col index for each lr above
    fig, ax = plt.subplots(figsize=(7, 4.6))
    sty = {"SC × LoRA": ("-o", C_OURS), "SC × fullft": ("--s", C_BASE),
           "GCv1 × LoRA": ("-o", "#2ca02c"), "GCv1 × fullft": ("--s", C_ALT)}
    for r in rows:
        name = r[0].replace("**", "").split(" (ST")[0].strip()
        xv, yv = [], []
        for lr, ci in zip(lrs, colmap):
            cell = r[ci].strip()
            if cell not in ("—", "-", ""):
                xv.append(lr); yv.append(num(cell))
        st, col = sty.get(name, ("-o", "#999"))
        # connect each series' own measured LRs (LoRA: 3 pts, fullft: 4 pts)
        ax.plot(xv, yv, st, color=col, lw=2, ms=7, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Encoder learning rate (log scale)")
    ax.set_ylabel("ARI (%, n=5)")
    ax.axvline(4e-5, color="k", ls=":", lw=1)
    ax.text(4e-5, ax.get_ylim()[0], " canonical", fontsize=8, rotation=90,
            va="bottom")
    ax.set_title("LoRA is LR-robust; full fine-tuning is not\n"
                 "(shared ST-SoftIdent + featrec=1.5 scaffold)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both")
    _save(fig, "fig_lora_lr_swing")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def main():
    # delete stale oral-era figures
    for p in list(FIG.glob("fig_*.pdf")) + list(FIG.glob("fig_*.png")):
        p.unlink()
    grid = (ELEC / "v2_grid_summary.md").read_text()
    # default points come from the GCv1×MOVi-C rescue-headline cell
    gc_movic = _grid_block(grid, "## Grounded Correspondence", "MOVi-C")
    dflt = gc_movic.get("st", np.nan)

    fig_rescue_forest()
    fig_predictor_ablation()
    _ablation("### featrec weight", "w=1.5 (default)", dflt,
              "featrec self-distillation weight", "fig_featrec_ablation",
              "Feature-reconstruction weight sweep (GCv1 × MOVi-C)")
    _ablation("### LoRA rank", "r=8 (default)", dflt,
              "LoRA rank", "fig_lora_rank_ablation",
              "LoRA rank sweep (GCv1 × MOVi-C)")
    _ablation("### R3 unfreeze step", "step 5000 (default)", dflt,
              "LoRA unfreeze step", "fig_unfreeze_ablation",
              "LoRA unfreeze-step sweep (GCv1 × MOVi-C)")
    fig_d1light()
    s1 = _traj_fig("val/ari", "Validation ARI (%)",
                   "Convergence", "fig_convergence", scale=100.0)
    s2 = _traj_fig("train/diag/feature_pr", "Backbone feature participation ratio",
                   "Feature-rank recovery", "fig_feature_pr")
    s3 = _traj_fig("train/predictor_rel_change",
                   "Predictor relative change",
                   "Predictor drift dynamics", "fig_drift_dynamics")
    fig_lora_lr_swing()

    (FIG / "FIGURE_NOTES.md").write_text(
        "# Figure provenance\n\n"
        "All quantitative figures regenerated from v7.5 sources by "
        "`scripts/tpami_make_figures.py` with the single canonical naming "
        "(STYLE_GUIDE §2). They supersede the stale oral-era `fig_*` "
        "(which had n=3 numbers + a different naming scheme).\n\n"
        "Aggregate figures (forest, predictor ablation, featrec/rank/"
        "unfreeze sweeps, D1-light): n=5, from paired_stats.md / "
        "v2_grid_summary.md.\n\n"
        "Trajectory figures (convergence, feature-rank, drift): single "
        "representative run per method (SlotContrast × MOVi-D), seeds: "
        + "; ".join(s1) + ". Per-step from <cell>/metrics/version_0/"
        "metrics.csv. Use only for the qualitative trajectory shape; the "
        "n=5 endpoint statistics are in the tables.\n\n"
        "fig_lora_lr_swing: §8.2.7 scaffolded LoRA-LR sweep (n=5).\n")
    print("wrote figures/FIGURE_NOTES.md")


if __name__ == "__main__":
    main()
