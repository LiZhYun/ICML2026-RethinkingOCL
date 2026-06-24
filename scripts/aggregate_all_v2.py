"""
Aggregate every 100K v2-grid run into a single per-cell summary.

Walks /scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k/slotcontrast_v2_100k/,
parses each run's metrics.csv, dedupes by (cell, seed) keeping the run with the
highest reached val step, then groups by experimental cell and emits:
  1. A compact CSV with per-cell mean/std for every val metric.
  2. A markdown table per dataset × architecture, ready to drop into a report.
Also handles the cross-backbone grid as a separate table.
"""

from __future__ import annotations
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

V2_ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k/slotcontrast_v2_100k"
)
XBB_ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_cross_backbone/v1_100k/"
    "slotcontrast_cross_backbone"
)
OUT_DIR = Path("/scratch/elec/t41020_egovla")

# v2 naming: <dataset>_<arch>_<variant>_v2_s<seed>
V2_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<dataset>movic|movid|movie|ytvis)_"
    r"(?P<arch>sc|gcv1_pf)_"
    r"(?P<variant>[a-zA-Z0-9_]+?)"
    r"_v2_s(?P<seed>\d+)$"
)

# cross-backbone naming: ytvis_sc_<bb>_<variant>_s<seed>
XBB_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"ytvis_sc_(?P<bb>dinov[123])_(?P<variant>learned|identity)_s(?P<seed>\d+)$"
)

METRIC_KEYS = ("val/ari", "val/fg_ari", "val/mbo", "val/image_ari", "val/image_mbo")


def parse_final_val(csv_path: Path) -> tuple[int, dict[str, float]] | None:
    """Return (highest_val_step, last_val_row_metrics) or None if no val row."""
    last_val_row = None
    last_val_step = -1
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                ari = row.get("val/ari", "").strip()
                if not ari:
                    continue
                step_str = row.get("step", "").strip()
                step = int(step_str) if step_str else -1
                if step >= last_val_step:
                    last_val_step = step
                    last_val_row = row
    except (OSError, ValueError):
        return None
    if last_val_row is None:
        return None
    parsed = {}
    for k in METRIC_KEYS:
        v = last_val_row.get(k, "").strip()
        parsed[k] = float(v) if v else float("nan")
    return last_val_step, parsed


def fmt_cell(vals: list[float]) -> str:
    """Mean ± std on the %-scale (0-100). Returns dash if empty."""
    vals = [v for v in vals if v == v]  # NaN filter
    if not vals:
        return "—"
    vals_pct = [v * 100 for v in vals]
    mean = statistics.mean(vals_pct)
    std = statistics.stdev(vals_pct) if len(vals_pct) >= 2 else 0.0
    return f"{mean:5.2f} $\\pm$ {std:4.2f}" if std > 0 else f"{mean:5.2f}"


# --------------------------------------------------------------------- v2 grid

def aggregate_v2() -> dict:
    # (dataset, arch, variant) -> seed -> (step, metrics)
    best: dict[tuple[str, str, str], dict[int, tuple[int, dict]]] = defaultdict(dict)
    skipped_no_match = 0
    skipped_no_val = 0

    for d in sorted(V2_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = V2_RE.match(d.name)
        if not m:
            skipped_no_match += 1
            continue
        ds, arch, variant, seed = (
            m.group("dataset"),
            m.group("arch"),
            m.group("variant"),
            int(m.group("seed")),
        )
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val(csv_path)
        if res is None:
            skipped_no_val += 1
            continue
        step, vals = res
        cell = (ds, arch, variant)
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)

    print(f"  v2: skipped {skipped_no_match} unparsable dirs, "
          f"{skipped_no_val} with no val row")
    return best


def aggregate_xbb() -> dict:
    best: dict[tuple[str, str], dict[int, tuple[int, dict]]] = defaultdict(dict)
    for d in sorted(XBB_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = XBB_RE.match(d.name)
        if not m:
            continue
        bb = m.group("bb")
        variant = m.group("variant")
        seed = int(m.group("seed"))
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val(csv_path)
        if res is None:
            continue
        step, vals = res
        cell = (bb, variant)
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)
    return best


# --------------------------------------------------------------------- output

def write_per_cell_csv(v2_best: dict, xbb_best: dict, path: Path) -> None:
    with path.open("w") as f:
        w = csv.writer(f)
        w.writerow(
            ["grid", "dataset_or_bb", "arch_or_predictor", "variant",
             "n_seeds", "max_step",
             "ari_mean", "ari_std", "fg_ari_mean", "fg_ari_std",
             "mbo_mean", "mbo_std",
             "image_ari_mean", "image_ari_std",
             "image_mbo_mean", "image_mbo_std"]
        )
        # v2
        for (ds, arch, var), seeds in sorted(v2_best.items()):
            seed_metrics = list(seeds.values())
            n = len(seed_metrics)
            max_step = max(s for s, _ in seed_metrics)
            row = ["v2", ds, arch, var, n, max_step]
            for k in METRIC_KEYS:
                vals = [v[k] for _, v in seed_metrics if v[k] == v[k]]
                vals_pct = [v * 100 for v in vals]
                mean = statistics.mean(vals_pct) if vals_pct else float("nan")
                std = statistics.stdev(vals_pct) if len(vals_pct) >= 2 else 0.0
                row += [f"{mean:.4f}" if mean == mean else "",
                        f"{std:.4f}"]
            w.writerow(row)
        # xbb
        for (bb, var), seeds in sorted(xbb_best.items()):
            seed_metrics = list(seeds.values())
            n = len(seed_metrics)
            max_step = max(s for s, _ in seed_metrics)
            row = ["xbb", bb, "sc", var, n, max_step]
            for k in METRIC_KEYS:
                vals = [v[k] for _, v in seed_metrics if v[k] == v[k]]
                vals_pct = [v * 100 for v in vals]
                mean = statistics.mean(vals_pct) if vals_pct else float("nan")
                std = statistics.stdev(vals_pct) if len(vals_pct) >= 2 else 0.0
                row += [f"{mean:.4f}" if mean == mean else "",
                        f"{std:.4f}"]
            w.writerow(row)


def md_v2_grid(v2_best: dict, path: Path) -> None:
    DATASETS = [
        ("movic", "MOVi-C"),
        ("movid", "MOVi-D"),
        ("movie", "MOVi-E"),
        ("ytvis", "YouTube-VIS 2021"),
    ]
    ARCHES = [("sc", "SlotContrast"), ("gcv1_pf", "Grounded Correspondence (per-frame init)")]
    CORE_VARIANTS = [
        ("frozen", "Frozen backbone (no LoRA, no fine-tune)"),
        ("default", "Default LoRA rescue (ST-SoftIdent + LoRA r=8 + featrec=1.5)"),
        ("st", "ST-SoftIdent + LoRA + featrec=1.5 (rescue headline)"),
        ("noST", "Soft SoftIdent (no straight-through) + LoRA + featrec=1.5"),
        ("hung", "Hungarian predictor + LoRA + featrec=1.5"),
        ("nomatch", "No predictor matching (skip_predictor=true)"),
        ("featrec0", "Default rescue with featrec=0 (loss ablation)"),
        ("fullft", "Full fine-tune (no LoRA) + ST + featrec=1.5"),
    ]
    D1_VARIANTS = [
        ("d1_default", "D1-light + default rescue"),
        ("d1_st", "D1-light + ST rescue"),
        ("d1_nomatch", "D1-light + nomatch"),
    ]

    with path.open("w") as f:
        f.write("# v2 grid — 100k-step main results\n\n")
        for arch_key, arch_name in ARCHES:
            f.write(f"## {arch_name}\n\n")
            for ds_key, ds_name in DATASETS:
                f.write(f"### {ds_name}\n\n")
                f.write("| Variant | n | step | ARI | FG-ARI | mBO | img-ARI | img-mBO |\n")
                f.write("|---|---:|---:|---|---|---|---|---|\n")
                for var_key, var_label in CORE_VARIANTS + D1_VARIANTS:
                    seeds = v2_best.get((ds_key, arch_key, var_key), {})
                    if not seeds:
                        continue
                    seed_metrics = list(seeds.values())
                    n = len(seed_metrics)
                    max_step = max(s for s, _ in seed_metrics)
                    cells = []
                    for k in METRIC_KEYS:
                        vals = [v[k] for _, v in seed_metrics]
                        cells.append(fmt_cell(vals).replace("$\\pm$", "±"))
                    f.write(f"| {var_label} | {n} | {max_step} | "
                            + " | ".join(cells) + " |\n")
                f.write("\n")

        # Ablation cells (movic + gcv1 only)
        f.write("## Ablation sweeps (MOVi-C × GCv1)\n\n")
        ABL_GROUPS = [
            ("R3 unfreeze step", [
                ("abl_r3_0", "step 0 (immediate)"),
                ("abl_r3_2500", "step 2500"),
                ("abl_r3_10k", "step 10000"),
            ]),
            ("LoRA rank", [
                ("abl_lora_r4", "r=4"),
                ("abl_lora_r16", "r=16"),
                ("abl_lora_r32", "r=32"),
            ]),
            ("featrec weight", [
                ("abl_fw0p5", "w=0.5"),
                ("abl_fw1p0", "w=1.0"),
                ("abl_fw2p0", "w=2.0"),
            ]),
        ]
        for group_name, items in ABL_GROUPS:
            f.write(f"### {group_name}\n\n")
            f.write("| Variant | n | step | ARI | FG-ARI | mBO | img-ARI | img-mBO |\n")
            f.write("|---|---:|---:|---|---|---|---|---|\n")
            for var_key, var_label in items:
                seeds = v2_best.get(("movic", "gcv1_pf", var_key), {})
                if not seeds:
                    continue
                seed_metrics = list(seeds.values())
                n = len(seed_metrics)
                max_step = max(s for s, _ in seed_metrics)
                cells = []
                for k in METRIC_KEYS:
                    vals = [v[k] for _, v in seed_metrics]
                    cells.append(fmt_cell(vals).replace("$\\pm$", "±"))
                f.write(f"| {var_label} | {n} | {max_step} | "
                        + " | ".join(cells) + " |\n")
            f.write("\n")


def md_xbb(xbb_best: dict, path: Path) -> None:
    with path.open("w") as f:
        f.write("# Cross-backbone predictor ablation (SC × YT-VIS, n=3)\n\n")
        f.write("| Backbone | Variant | n | ARI | FG-ARI | mBO |\n")
        f.write("|---|---|---:|---|---|---|\n")
        for bb in ("dinov1", "dinov2", "dinov3"):
            bb_name = {"dinov1": "DINOv1", "dinov2": "DINOv2", "dinov3": "DINOv3"}[bb]
            for var in ("identity", "learned"):
                seeds = xbb_best.get((bb, var), {})
                if not seeds:
                    continue
                seed_metrics = list(seeds.values())
                n = len(seed_metrics)
                cells = []
                for k in ("val/ari", "val/fg_ari", "val/mbo"):
                    vals = [v[k] for _, v in seed_metrics]
                    cells.append(fmt_cell(vals).replace("$\\pm$", "±"))
                f.write(f"| {bb_name} | {var} | {n} | " + " | ".join(cells) + " |\n")
        f.write("\n")


def main() -> None:
    print("Parsing v2 grid...")
    v2 = aggregate_v2()
    print(f"  v2 cells: {len(v2)}")
    print("Parsing cross-backbone...")
    xbb = aggregate_xbb()
    print(f"  xbb cells: {len(xbb)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "all_results_per_cell.csv"
    md_v2 = OUT_DIR / "v2_grid_summary.md"
    md_x = OUT_DIR / "cross_backbone_summary.md"

    write_per_cell_csv(v2, xbb, csv_path)
    md_v2_grid(v2, md_v2)
    md_xbb(xbb, md_x)

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_v2}")
    print(f"Wrote: {md_x}")
    print("\nTop-10 cells by n_seeds:")
    rows = sorted(
        ((len(s), ds, a, v) for (ds, a, v), s in v2.items()),
        reverse=True,
    )
    for n, ds, a, v in rows[:20]:
        print(f"  n={n}  {ds}/{a}/{v}")


if __name__ == "__main__":
    main()
