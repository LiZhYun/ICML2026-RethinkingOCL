"""Figures + LaTeX tables for the 2026-05-29 breadth campaign → tpami-bundle/added_exps/.

Self-contained: does NOT modify the main bundle figures/tables. Reads
 - tpami-bundle/added_exps/added_exps_per_cell.csv  (the 4 new waves)
 - tpami-bundle/data/all_results_per_cell.csv        (v2 frozen + st baselines)
 - the elec phase_a/phase_b run dirs                 (pre-existing MOVi-D last-k /
                                                       LoRA-LR + MAE-identity, to
                                                       complete the 4-dataset tables)
and writes, into tpami-bundle/added_exps/:
  figures/fig_frozen_nomatch_factorial.{svg,pdf,png}
  figures/fig_lora_lr_robustness.{svg,pdf,png}
  tables/{tab_frozen_nomatch,tab_lastk_breadth,tab_lora_lr_breadth,tab_mae_identity_learned}.tex

Nature style (Arial 7pt, editable SVG, NMI pastel), matching figures_nature/.
Numbers only; no interpretation asserted. Run on a node with the slotcontrast env.
"""
import csv
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False, "savefig.dpi": 600, "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
})

REPO = Path(__file__).resolve().parent.parent
BUNDLE = REPO / "tpami-bundle"
AE = BUNDLE / "added_exps"
FIG = AE / "figures"; TAB = AE / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
ELEC = Path("/scratch/elec/t41020_egovla")
PA = ELEC / "slotcontrast_phase_a/v1_100k/slotcontrast_phase_a"
PB = ELEC / "slotcontrast_phase_b/v1_100k/slotcontrast_phase_b"

MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM
NMI = {"grey": "#808080", "hero": "#D85684",
       "movic": "#7884B4", "movid": "#D85684", "movie": "#52A0C8", "ytvis": "#E0A030"}
DSN = {"movic": "MOVi-C", "movid": "MOVi-D", "movie": "MOVi-E", "ytvis": "YT-VIS"}
DSO = ["movic", "movid", "movie", "ytvis"]
ARCHN = {"sc": "SlotContrast", "gcv1_pf": "Grounded Correspondence"}


def final_ari(mc):
    last = (-1, None)
    try:
        for row in csv.DictReader(open(mc)):
            if row.get("val/ari", "").strip():
                s = int(row["step"]) if row.get("step", "").strip() else -1
                if s >= last[0]:
                    last = (s, {k: (float(row[k]) if row.get(k, "").strip() else float("nan"))
                               for k in ("val/ari", "val/fg_ari", "val/mbo")})
    except OSError:
        return None
    return last[1]


def agg_root(root, rgx):
    """{groupkey: {metric: (mean%, std%)}} aggregated over seeds."""
    cells = {}
    if root.exists():
        for d in sorted(root.iterdir()):
            m = rgx.match(d.name)
            if not m:
                continue
            v = final_ari(d / "metrics" / "version_0" / "metrics.csv")
            if v is None:
                continue
            cells.setdefault(m.groups(), []).append(v)
    out = {}
    for k, vs in cells.items():
        out[k] = {}
        for met in ("val/ari", "val/fg_ari", "val/mbo"):
            xs = [x[met] * 100 for x in vs if x[met] == x[met]]
            out[k] = out.get(k, {})
            out[k][met] = (statistics.mean(xs), statistics.stdev(xs) if len(xs) >= 2 else 0.0) if xs else (float("nan"), 0.0)
    return out


# ---- load sources ----
added = list(csv.DictReader(open(AE / "added_exps_per_cell.csv")))
v2 = list(csv.DictReader(open(BUNDLE / "data" / "all_results_per_cell.csv")))


def added_cell(wave, ds, arch, variant):
    for r in added:
        if (r["wave"], r["dataset"], r["arch"], r["variant"]) == (wave, ds, arch, variant):
            return float(r["ari_mean"]), float(r["ari_std"])
    return None


def v2_cell(ds, arch, variant):
    for r in v2:
        if r["grid"] == "v2" and r["dataset_or_bb"] == ds and r["arch_or_predictor"] == arch and r["variant"] == variant:
            return float(r["ari_mean"]), float(r["ari_std"])
    return None


TS = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
movid_lastk = agg_root(PB, re.compile(TS + r"phaseB_movid_(?P<arch>sc|gcv1_pf)_(?P<v>lastk2|lastk4)_s\d+$"))
movid_lora = agg_root(PA, re.compile(TS + r"phaseA_movid_(?P<arch>sc|gcv1_pf)_lora_st_fr15_(?P<lr>lr4e-6|lr4e-4)_s\d+$"))
mae = agg_root(PA, re.compile(TS + r"phaseA_ytvis_sc_mae_native224_(?P<v>identity|learned)_s\d+$"))


# ================= FIGURE 1: frozen_nomatch factorial =================
def fig_factorial():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM), sharey=True)
    for ax, arch, lab in zip(axes, ("sc", "gcv1_pf"), ("a", "b")):
        x = np.arange(len(DSO)); w = 0.38
        fz = [v2_cell(d, arch, "frozen") for d in DSO]
        fn = [added_cell("frozen_nomatch", d, arch, "frozen_nomatch") for d in DSO]
        ax.bar(x - w / 2, [c[0] for c in fz], w, yerr=[c[1] for c in fz],
               color=NMI["grey"], edgecolor="white", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.6), label="Frozen (mechanism on)")
        ax.bar(x + w / 2, [c[0] for c in fn], w, yerr=[c[1] for c in fn],
               color=NMI["hero"], edgecolor="white", linewidth=0.4,
               error_kw=dict(lw=0.6, capsize=1.6), label="Frozen + mechanism off")
        ax.set_xticks(x); ax.set_xticklabels([DSN[d] for d in DSO])
        ax.set_title(ARCHN[arch], pad=4); ax.tick_params(length=2.2, pad=2)
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 85)
    axes[0].set_ylabel("ARI (%, n=5)")
    axes[0].legend(loc="lower left", handlelength=1.2, borderaxespad=0.5,
                   handletextpad=0.5, labelspacing=0.3)
    fig.suptitle("Frozen backbone with vs without the temporal mechanism "
                 "(SC: identity propagation; GC: no correspondence)",
                 fontsize=7.5, y=1.02)
    _save(fig, "fig_frozen_nomatch_factorial")


# ============== FIGURE 2: LoRA-LR robustness across datasets ==============
def fig_lr_robustness():
    # 3 points per (ds,arch): 0.1x, 1x (v2 st), 10x.  MOVi: 4e-6/4e-4; YT-VIS: 8e-6/8e-4.
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM), sharey=True)
    xs = np.array([0, 1, 2])
    for ax, arch, lab in zip(axes, ("sc", "gcv1_pf"), ("a", "b")):
        for d in DSO:
            lo_lr, hi_lr = ("lr8e-6", "lr8e-4") if d == "ytvis" else ("lr4e-6", "lr4e-4")
            if d == "movid":
                lo = movid_lora.get((arch, lo_lr), {}).get("val/ari")
                hi = movid_lora.get((arch, hi_lr), {}).get("val/ari")
            else:
                lo = added_cell("lora_lr", d, arch, lo_lr)
                hi = added_cell("lora_lr", d, arch, hi_lr)
            mid = v2_cell(d, arch, "st")
            if not (lo and hi and mid):
                continue
            ys = [lo[0], mid[0], hi[0]]; es = [lo[1], mid[1], hi[1]]
            ax.errorbar(xs, ys, yerr=es, fmt="-o", color=NMI[d], lw=1.1, ms=3.5,
                        capsize=1.6, elinewidth=0.6, label=DSN[d])
        ax.set_xticks(xs); ax.set_xticklabels(["0.1$\\times$", "1$\\times$\n(canon.)", "10$\\times$"])
        ax.set_xlabel("LoRA encoder LR (rel. to dataset canonical)")
        ax.set_title(ARCHN[arch], pad=4); ax.tick_params(length=2.2, pad=2)
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 90)
    axes[0].set_ylabel("ARI (%, n=5)")
    axes[1].legend(loc="lower left", handlelength=1.4, borderaxespad=0.5,
                   handletextpad=0.5, labelspacing=0.25, ncol=2, columnspacing=1.0)
    fig.suptitle("LoRA learning-rate robustness across datasets "
                 "(0.1$\\times$–10$\\times$ of each dataset's canonical LoRA LR)",
                 fontsize=7.5, y=1.02)
    _save(fig, "fig_lora_lr_robustness")


def _save(fig, name):
    fig.tight_layout(pad=0.7)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote added_exps/figures/{name}.{{svg,pdf,png}}")


# ================= TABLES (.tex, IEEEtran booktabs) =================
def _cell(t):
    return "--" if t is None else (f"{t[0]:.2f} $\\pm$ {t[1]:.2f}")


def tab_frozen_nomatch():
    rows = []
    for arch in ("sc", "gcv1_pf"):
        for d in DSO:
            fz = v2_cell(d, arch, "frozen"); fn = added_cell("frozen_nomatch", d, arch, "frozen_nomatch")
            delta = f"{fn[0]-fz[0]:+.2f}" if (fz and fn) else "--"
            rows.append(f"{ARCHN[arch]} & {DSN[d]} & {_cell(fz)} & {_cell(fn)} & {delta} \\\\")
    body = "\n".join(rows)
    (TAB / "tab_frozen_nomatch.tex").write_text(
        "% Auto-generated: frozen vs frozen_nomatch factorial control (ARI %, n=5)\n"
        "\\begin{table}[t]\\centering\\small\n"
        "\\caption{\\textbf{Frozen-backbone temporal-mechanism control ($n{=}5$).} "
        "frozen $=$ frozen backbone $+$ default predictor/matching; "
        "frozen\\_nomatch $=$ frozen backbone $+$ \\texttt{skip\\_predictor} "
        "(SC: identity propagation; GC: no correspondence). "
        "$\\Delta=$ frozen\\_nomatch $-$ frozen (ARI).}\n"
        "\\label{tab:frozen_nomatch}\n"
        "\\begin{tabular}{llccc}\\toprule\n"
        "Arch & Dataset & frozen & frozen\\_nomatch & $\\Delta$ ARI \\\\\\midrule\n"
        f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")
    print("wrote added_exps/tables/tab_frozen_nomatch.tex")


def tab_lastk():
    rows = []
    for arch in ("sc", "gcv1_pf"):
        for d in DSO:
            if d == "movid":
                k2 = movid_lastk.get((arch, "lastk2"), {}).get("val/ari")
                k4 = movid_lastk.get((arch, "lastk4"), {}).get("val/ari")
            else:
                k2 = added_cell("lastk", d, arch, "lastk2")
                k4 = added_cell("lastk", d, arch, "lastk4")
            tag = "" if d != "movid" else "$^\\dagger$"
            rows.append(f"{ARCHN[arch]} & {DSN[d]}{tag} & {_cell(k2)} & {_cell(k4)} \\\\")
    body = "\n".join(rows)
    (TAB / "tab_lastk_breadth.tex").write_text(
        "% Auto-generated: last-k partial-unfreeze breadth (ARI %, n=5). dagger = pre-existing MOVi-D (T3.2).\n"
        "\\begin{table}[t]\\centering\\small\n"
        "\\caption{\\textbf{Last-$k$ partial-unfreeze across all four datasets ($n{=}5$).} "
        "ST-SoftIdent $+$ featrec rescue with the final $k$ ViT blocks unfrozen "
        "(LoRA replaced). $^\\dagger$MOVi-D is the pre-existing T3.2 cell.}\n"
        "\\label{tab:lastk_breadth}\n"
        "\\begin{tabular}{llcc}\\toprule\n"
        "Arch & Dataset & last-2 & last-4 \\\\\\midrule\n"
        f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")
    print("wrote added_exps/tables/tab_lastk_breadth.tex")


def tab_lora_lr():
    rows = []
    for arch in ("sc", "gcv1_pf"):
        for d in DSO:
            lo_lr, hi_lr = ("lr8e-6", "lr8e-4") if d == "ytvis" else ("lr4e-6", "lr4e-4")
            if d == "movid":
                lo = movid_lora.get((arch, lo_lr), {}).get("val/ari")
                hi = movid_lora.get((arch, hi_lr), {}).get("val/ari")
            else:
                lo = added_cell("lora_lr", d, arch, lo_lr)
                hi = added_cell("lora_lr", d, arch, hi_lr)
            mid = v2_cell(d, arch, "st")
            tag = "$^\\dagger$" if d == "movid" else ""
            rows.append(f"{ARCHN[arch]} & {DSN[d]}{tag} & {_cell(lo)} & {_cell(mid)} & {_cell(hi)} \\\\")
    body = "\n".join(rows)
    (TAB / "tab_lora_lr_breadth.tex").write_text(
        "% Auto-generated: LoRA-LR robustness breadth (ARI %, n=5). 1x = v2 st cell. dagger=pre-existing MOVi-D (T1.4).\n"
        "\\begin{table}[t]\\centering\\small\n"
        "\\caption{\\textbf{LoRA encoder learning-rate robustness across all four "
        "datasets ($n{=}5$).} ARI at 0.1$\\times$, 1$\\times$ (canonical, from the "
        "main grid), and 10$\\times$ each dataset's canonical LoRA LR (MOVi: "
        "4e-5; YT-VIS: 8e-5). $^\\dagger$MOVi-D is the pre-existing T1.4 cell.}\n"
        "\\label{tab:lora_lr_breadth}\n"
        "\\begin{tabular}{llccc}\\toprule\n"
        "Arch & Dataset & 0.1$\\times$ & 1$\\times$ & 10$\\times$ \\\\\\midrule\n"
        f"{body}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")
    print("wrote added_exps/tables/tab_lora_lr_breadth.tex")


def tab_mae():
    idn = mae.get(("identity",), {}); lrn = mae.get(("learned",), {})
    def c(d, k):
        return _cell(d[k]) if d and k in d else "--"
    row = (f"MAE-ViT-B/16 & identity & {c(idn,'val/ari')} & {c(idn,'val/fg_ari')} & {c(idn,'val/mbo')} \\\\\n"
           f" & learned & {c(lrn,'val/ari')} & {c(lrn,'val/fg_ari')} & {c(lrn,'val/mbo')} \\\\")
    (TAB / "tab_mae_identity_learned.tex").write_text(
        "% Auto-generated: MAE identity vs learned predictor (SC x YT-VIS, n=3).\n"
        "\\begin{table}[t]\\centering\\small\n"
        "\\caption{\\textbf{MAE-ViT-B/16 backbone: identity vs learned predictor "
        "(SlotContrast $\\times$ YT-VIS, $n{=}3$).} Completes the cross-backbone "
        "identity-vs-learned comparison for the MAE (masked-image-modelling) "
        "paradigm. \\emph{Both} variants collapse to the documented degenerate "
        "one-slot solution (mBO $\\approx$ 6--10\\% vs $\\approx$27\\% for "
        "non-collapsed models; seeds are near-identical, hence $\\pm$0.00). The "
        "identity$\\approx$learned equivalence holds here because both collapse, "
        "not because both segment well \\textemdash\\ i.e. the MAE collapse is "
        "predictor-independent.}\n"
        "\\label{tab:mae_identity_learned}\n"
        "\\begin{tabular}{llccc}\\toprule\n"
        "Backbone & Predictor & ARI & FG-ARI & mBO \\\\\\midrule\n"
        f"{row}\n\\bottomrule\\end{{tabular}}\\end{{table}}\n")
    print("wrote added_exps/tables/tab_mae_identity_learned.tex")


def main():
    fig_factorial()
    fig_lr_robustness()
    tab_frozen_nomatch()
    tab_lastk()
    tab_lora_lr()
    tab_mae()
    print("done")


if __name__ == "__main__":
    main()
