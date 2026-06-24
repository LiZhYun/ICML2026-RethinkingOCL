"""Generate tpami-bundle/tables/tables.tex purely from the authoritative
source files (no hand-transcription). Run anytime the numbers change.

Sources:
  /scratch/elec/t41020_egovla/v2_grid_summary.md     (main grid)
  /scratch/elec/t41020_egovla/paired_stats.md        (F1-F9 paired-bootstrap)
  /scratch/elec/t41020_egovla/cross_backbone_summary.md
  EXPERIMENTAL_RESULTS.md §4.7 (k=18 F6) + §8.2.7 (LoRA-LR sweep)
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ELEC = Path("/scratch/elec/t41020_egovla")
OUT = REPO / "tpami-bundle" / "tables" / "tables.tex"

TEX = lambda s: (s.replace("±", r"$\pm$").replace("−", "$-$")
                  .replace("×10⁻⁵", r"\times10^{-5}").replace("×10⁻⁶", r"\times10^{-6}")
                  .replace("×10⁻⁴", r"\times10^{-4}").replace("≡", r"$\equiv$")
                  .replace("≤", r"$\leq$").replace("e-0", r"e{-}0").replace("e-", r"e{-}")
                  .replace("**", "").replace("%", r"\%").replace("&", r"\&")
                  .replace("_", r"\_"))


def md_rows(text, header_re, n_cols):
    """Data rows of the first markdown table appearing AFTER the first line
    matching header_re. Robust to the blank line between header and table:
    starts collecting at the first '|' line, stops at the first non-'|'
    line once collection has begun."""
    m = re.search(header_re, text)
    if not m:
        return []
    rows, started = [], False
    for line in text[m.end():].splitlines():
        s = line.strip()
        if s.startswith("|"):
            started = True
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) < n_cols:
                continue
            if set("".join(cells)) <= set("-: "):           # separator row
                continue
            if cells[0].lower() in ("comparison", "variant", "dataset",
                                    "method", "sorted by raw p", "cell",
                                    "arch", "backbone"):
                continue
            rows.append(cells)
        elif started and s == "":
            continue                                          # blank inside ok
        elif started:
            break                                             # table ended
    return rows


def fam_table(ps, fam_header, caption, label):
    rows = md_rows(ps, re.escape(fam_header), 7)
    body = []
    for r in rows:
        comp, n, dlt, ci, rawp, holm, rej = r[:7]
        comp = comp.split(":")[0].strip()
        rej = "yes" in rej.lower()
        holm_fmt = (r"\textbf{%s}" % TEX(holm)) if rej else TEX(holm)
        body.append(f"{TEX(comp)} & {TEX(dlt)} & {TEX(ci)} & {TEX(rawp)} & {holm_fmt} & "
                     + ("\\textbf{yes}" if rej else "no") + r" \\")
    return rf"""
\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\begin{{tabular}}{{l r c r r c}}
\toprule
Comparison & $\Delta$ & 95\% CI & raw $p$ & Holm $p$ & reject? \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    ps = (ELEC / "paired_stats.md").read_text()
    er = (REPO / "EXPERIMENTAL_RESULTS.md").read_text()
    grid = (ELEC / "v2_grid_summary.md").read_text()
    cb = (ELEC / "cross_backbone_summary.md").read_text()

    parts = [r"""% ============================================================================
% TPAMI tables — AUTO-GENERATED from source by scripts/tpami_make_tables.py.
% Do not hand-edit numbers here; edit the source files and regenerate.
% Sources: v2_grid_summary.md, paired_stats.md, cross_backbone_summary.md,
%          EXPERIMENTAL_RESULTS.md (§4.7 k=18 F6, §8.2.7 LoRA-LR sweep).
% Requires: \usepackage{booktabs}
% ============================================================================
"""]

    # ---- Main grid (SC + GCv1, frozen / rescue / nomatch) -------------------
    def grid_block(arch_header, ds):
        # isolate the architecture block (## SlotContrast .. next ## )
        gi = grid.find(arch_header)
        if gi < 0:
            return {}
        seg = grid[gi + len(arch_header):]
        nxt = re.search(r"\n## ", seg)
        seg = seg[: nxt.start()] if nxt else seg
        # isolate the dataset sub-block (### <ds> .. next ### )
        di = seg.find(f"### {ds}")
        if di < 0:
            return {}
        sub = seg[di + len(f"### {ds}"):]
        nxt = re.search(r"\n### ", sub)
        sub = sub[: nxt.start()] if nxt else sub
        out = {}
        for line in sub.splitlines():
            s = line.strip()
            if not s.startswith("|"):
                continue
            c = [x.strip() for x in s.strip("|").split("|")]
            if len(c) < 5 or set("".join(c)) <= set("-: "):
                continue
            key = c[0].lower()
            if "frozen backbone" in key:          out["frozen"] = c
            elif "rescue headline" in key:        out["st"] = c
            elif "no predictor matching" in key:  out["nomatch"] = c
        return out

    rows = []
    for ds_label, ds in [("MOVi-C", "MOVi-C"), ("MOVi-D", "MOVi-D"),
                          ("MOVi-E", "MOVi-E"), ("YT-VIS", "YouTube-VIS 2021")]:
        sc = grid_block("## SlotContrast", ds)
        gc = grid_block("## Grounded Correspondence", ds)
        for mi, metric in enumerate(["ARI", "FG-ARI", "mBO"]):
            col = {"ARI": 3, "FG-ARI": 4, "mBO": 5}[metric]
            def cell(d, k):
                return TEX(d[k][col]) if k in d and len(d[k]) > col else "---"
            rows.append(
                f"{ds_label if mi==0 else ''} & {metric} & "
                f"{cell(sc,'frozen')} & {cell(sc,'st')} & {cell(sc,'nomatch')} & & "
                f"{cell(gc,'frozen')} & {cell(gc,'st')} & {cell(gc,'nomatch')} \\\\")
        rows.append(r"\midrule")
    if rows and rows[-1] == r"\midrule":
        rows.pop()
    parts.append(r"""
\begin{table*}[t]
\centering
\caption{\textbf{Main grid headline.} Frozen backbone vs the rescue
protocol (Straight-Through SoftIdent + LoRA r=8 + featrec=1.5) vs
Identity propagation ($Q_t{=}S_{t-1}$); $n{=}5$ at 100K steps, \%
scale, mean $\pm$ std. For SlotContrast the rescue is Holm-tied with
Identity propagation (F2 0/12); for Grounded Correspondence the
predictor is load-bearing (F4 12/12).}
\label{tab:main_grid}
\small
\begin{tabular}{l l ccc c ccc}
\toprule
& & \multicolumn{3}{c}{\textbf{SlotContrast}} & & \multicolumn{3}{c}{\textbf{Grounded Correspondence}} \\
\cmidrule(lr){3-5}\cmidrule(lr){7-9}
Dataset & Metric & \shortstack{Frozen\\backbone} & \shortstack{Straight-Through\\SoftIdent} & \shortstack{Identity\\propagation} & & \shortstack{Frozen\\backbone} & \shortstack{Straight-Through\\SoftIdent} & \shortstack{Identity\\propagation} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

    # ---- F2, F4 from paired_stats.md ---------------------------------------
    parts.append(fam_table(
        ps, "### F2 — SC predictor ablation (st − nomatch), 12 tests",
        r"\textbf{F2 — SlotContrast predictor ablation is Holm-tied "
        r"($n{=}5$).} $\Delta=$ (Straight-Through SoftIdent) $-$ "
        r"(Identity propagation). 0/12 cells reject H$_0$ after Holm.",
        "tab:f2_sc"))
    parts.append(fam_table(
        ps, "### F4 — GCv1 predictor necessity (frozen − nomatch), 12 tests",
        r"\textbf{F4 — Grounded Correspondence predictor necessity "
        r"(decisive, $n{=}5$).} $\Delta=$ (Frozen backbone) $-$ "
        r"(Identity propagation). 12/12 Holm-significant.",
        "tab:f4_gcv1"))
    parts.append(fam_table(
        ps, "### F1 — SC rescue lift (st − frozen), 12 tests",
        r"\textbf{F1 — SlotContrast rescue lift over the frozen baseline "
        r"($n{=}5$).} $\Delta=$ (Straight-Through SoftIdent) $-$ "
        r"(Frozen backbone). 5/12 Holm-significant "
        r"(every MOVi-mBO plus MOVi-E ARI/FG-ARI).", "tab:f1_sc"))

    # ---- F6 k=18 from EXPERIMENTAL_RESULTS.md §4.7 -------------------------
    f6 = md_rows(er, r"\| Sorted by raw p \| Δ \(LoRA − fullft@1e-5\) \| raw p \| Holm p \(k=18\) \| reject \|", 5)
    f6b = []
    for r in f6:
        cell, dlt, rawp, holm, rej = r[:5]
        is_rej = "yes" in rej.lower()
        f6b.append(f"{TEX(cell)} & {TEX(dlt)} & {TEX(rawp)} & "
                   + ((r"\textbf{%s}" % TEX(holm)) if is_rej else TEX(holm))
                   + " & " + (r"\textbf{%s}" % TEX(rej)) + r" \\")
    parts.append(r"""
\begin{table}[t]
\centering
\caption{\textbf{F6 — LoRA vs Full fine-tuning at the tuned LR
($k{=}18$ unique tests).} $\Delta=$ LoRA $-$ Full fine-tuning
(lr$=10^{-5}$), sorted by raw $p$, Holm-corrected at $k{=}18$ (MOVi
ARI$\equiv$FG-ARI collapsed). LoRA wins 1 cell
(Grounded Correspondence$\times$MOVi-D$\times$ARI), Full fine-tuning
wins 2 (SlotContrast$\times$YT-VIS$\times$ARI;
Grounded Correspondence$\times$MOVi-E$\times$mBO), 15/18 Holm-tied.}
\label{tab:f6}
\small
\begin{tabular}{l r r r c}
\toprule
Cell & $\Delta$ & raw $p$ & Holm $p$ ($k{=}18$) & reject \\
\midrule
""" + "\n".join(f6b) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # ---- §8.2.7 LoRA-LR sweep ---------------------------------------------
    lr = md_rows(er, r"\| Method \| lr=1×10⁻⁵ \| lr=4×10⁻⁶ \| lr=4×10⁻⁵ \(canon\) \| lr=1×10⁻⁴ \| lr=4×10⁻⁴ \| LR-swing \|", 7)
    _lrname = {"SC × LoRA": "SC, LoRA", "SC × fullft": "SC, Full fine-tuning",
               "GCv1 × LoRA": "GCv1, LoRA",
               "GCv1 × fullft": "GCv1, Full fine-tuning"}
    lrb = []
    for r in lr:
        nm = r[0].replace("**", "").split(" (ST")[0].strip()
        cells = [_lrname.get(nm, nm)] + [TEX(x) for x in r[1:7]]
        lrb.append(" & ".join(cells) + r" \\")
    parts.append(r"""
\begin{table}[t]
\centering
\caption{\textbf{Scaffolded LoRA encoder-LR sweep ($n{=}5$, §8.2.7).}
Same ST-SoftIdent + featrec=1.5 scaffold; LoRA vs full fine-tune differ
in \textbf{LR-robustness}, not asymptotic performance. LoRA's canonical
LR ($4\times10^{-5}$) is the v2 default LR at which fullft collapses.}
\label{tab:lora_lr}
\small
\begin{tabular}{l rrrrr r}
\toprule
Method & $1{\times}10^{-5}$ & $4{\times}10^{-6}$ & $4{\times}10^{-5}$ & $1{\times}10^{-4}$ & $4{\times}10^{-4}$ & swing \\
\midrule
""" + "\n".join(lrb) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

    # ---- Cross-backbone failure-mode --------------------------------------
    cbr = md_rows(cb, r"\| Backbone \| Variant \| n \| ARI \| FG-ARI \| mBO \|", 6)
    cbb = []
    for r in cbr:
        bb, var, n, a, f, mb = r[:6]
        cbb.append(f"{TEX(bb)} & {TEX(var)} & {TEX(n)} & {TEX(a)} & {TEX(f)} & {TEX(mb)} \\\\")
    parts.append(r"""
\begin{table}[t]
\centering
\caption{\textbf{Cross-backbone failure-mode appendix (SC $\times$
YT-VIS, $n{=}3$).} Identity vs learned predictor. DINOv1/v2 Holm-tied
(F5 0/9; TOST on DINOv2 ARI $p_{\text{Holm}}{=}0.0044$). DINOv3 (and
MAE, see §5) collapse into one-large-slot; documented as failure modes,
not predictor-equivalence backbones.}
\label{tab:cross_backbone}
\small
\begin{tabular}{l l c c c c}
\toprule
Backbone & Predictor & $n$ & ARI & FG-ARI & mBO \\
\midrule
""" + "\n".join(cbb) + r"""
\bottomrule
\multicolumn{6}{l}{\footnotesize Predictor-equivalence scoped to DINOv1/v2; DINOv3+MAE are failure-mode entries (§5, §8.3).}
\end{tabular}
\end{table}
""")

    OUT.write_text("\n".join(parts))
    n_tab = sum(p.count(r"\begin{table") for p in parts)
    print(f"wrote {OUT} ({n_tab} tables, {len(''.join(parts))} chars)")


if __name__ == "__main__":
    main()
