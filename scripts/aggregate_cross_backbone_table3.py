"""
Aggregate per-backbone × predictor val metrics from the 18 cross-backbone runs
into a LaTeX table matching the structure of GC paper Table 3.

For each run directory under
  /scratch/elec/t41020_egovla/slotcontrast_cross_backbone/v1_100k/slotcontrast_cross_backbone/
parse metrics/version_0/metrics.csv, take the LAST val row (step 99999),
group by (backbone, variant), and report mean ± std over the 3 seeds for
val/ari, val/fg_ari, val/mbo.
"""

from __future__ import annotations
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_cross_backbone/v1_100k/"
    "slotcontrast_cross_backbone"
)

# experiment_name: ytvis_sc_<backbone>_<variant>_s<seed>
NAME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"ytvis_sc_(?P<bb>dinov[123])_(?P<variant>learned|identity)_s(?P<seed>\d+)"
)

METRIC_KEYS = ("val/ari", "val/fg_ari", "val/mbo")


def last_val_row(csv_path: Path) -> dict[str, float]:
    """Return the last row that has a non-empty val/ari field."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        last_val = None
        for row in reader:
            ari = row.get("val/ari", "").strip()
            if ari:
                last_val = row
    if last_val is None:
        raise RuntimeError(f"no val row found in {csv_path}")
    return {k: float(last_val[k]) for k in METRIC_KEYS}


def main() -> None:
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in METRIC_KEYS}
    )

    run_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if len(run_dirs) != 18:
        print(f"[warn] expected 18 run dirs, found {len(run_dirs)}")

    for d in run_dirs:
        m = NAME_RE.match(d.name)
        if not m:
            print(f"[skip] unrecognised dir: {d.name}")
            continue
        bb = m.group("bb")
        variant = m.group("variant")
        seed = int(m.group("seed"))

        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            print(f"[skip] missing metrics.csv for {d.name}")
            continue

        vals = last_val_row(csv_path)
        for k in METRIC_KEYS:
            by_cell[(bb, variant)][k].append(vals[k])
        print(
            f"{bb:7s} {variant:9s} s{seed}: "
            f"ARI={vals['val/ari']*100:6.2f}  "
            f"FG-ARI={vals['val/fg_ari']*100:6.2f}  "
            f"mBO={vals['val/mbo']*100:6.2f}"
        )

    # ---- aggregate ----
    print("\n" + "=" * 78)
    print("AGGREGATED  (% scale, mean ± std over n seeds)")
    print("=" * 78)
    print(f"{'backbone':8s} {'variant':10s} {'n':>3s}  {'ARI':>14s}  {'FG-ARI':>14s}  {'mBO':>14s}")

    def fmt(vals: list[float]) -> tuple[str, float, float]:
        vals_pct = [v * 100 for v in vals]
        mean = statistics.mean(vals_pct)
        std = statistics.stdev(vals_pct) if len(vals_pct) >= 2 else 0.0
        return f"{mean:5.2f} ± {std:4.2f}", mean, std

    rows_for_latex: list[tuple[str, str, dict[str, str]]] = []
    for bb in ("dinov1", "dinov2", "dinov3"):
        for variant in ("identity", "learned"):
            cell = by_cell.get((bb, variant), {})
            n = len(cell.get("val/ari", []))
            fmt_cells = {}
            for k in METRIC_KEYS:
                vals = cell.get(k, [])
                s, _, _ = fmt(vals)
                fmt_cells[k] = s
            print(
                f"{bb:8s} {variant:10s} {n:3d}  "
                f"{fmt_cells['val/ari']:>14s}  "
                f"{fmt_cells['val/fg_ari']:>14s}  "
                f"{fmt_cells['val/mbo']:>14s}"
            )
            rows_for_latex.append((bb, variant, fmt_cells))

    # ---- LaTeX (matches Table 3 structure but adds backbone column) ----
    print("\n" + "=" * 78)
    print("LATEX (matches Table 3 row layout, backbone added as group)")
    print("=" * 78)

    bb_pretty = {"dinov1": "DINOv1", "dinov2": "DINOv2", "dinov3": "DINOv3"}
    variant_pretty = {
        "identity": r"Identity ($Q_t = S_{t-1}$)",
        "learned": "Transformer encoder",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SlotContrast with and without learned temporal prediction on YouTube-VIS, "
        r"across DINOv1/v2/v3 backbones. The identity baseline $Q_t = S_{t-1}$ matches "
        r"the full model across all three pretrained backbones. "
        r"Mean and standard deviation over 3 seeds.}",
        r"\label{tab:cross_backbone_predictor}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Backbone & Temporal Module & ARI & FG-ARI & mBO \\",
        r"\midrule",
    ]

    # group identity above learned per backbone, with \cmidrule between backbones
    for i, bb in enumerate(("dinov1", "dinov2", "dinov3")):
        if i > 0:
            lines.append(r"\midrule")
        for j, variant in enumerate(("identity", "learned")):
            cells = next(c for (b, v, c) in rows_for_latex if b == bb and v == variant)
            bb_cell = rf"\multirow{{2}}{{*}}{{{bb_pretty[bb]}}}" if j == 0 else ""
            lines.append(
                f"{bb_cell} & {variant_pretty[variant]} & "
                f"{cells['val/ari']} & {cells['val/fg_ari']} & {cells['val/mbo']} \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    print(latex)

    out_path = (
        ROOT.parent / "table3_cross_backbone.tex"
    )
    out_path.write_text(latex + "\n")
    print(f"\nLaTeX written to: {out_path}")


if __name__ == "__main__":
    main()
