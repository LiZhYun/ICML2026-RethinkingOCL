"""Collect the 2026-05-29 breadth-campaign results into tpami-bundle/added_exps/.

Standalone (does NOT modify the paper's aggregators). Walks the three save
roots, parses each campaign cell's final-validation metrics with the SAME
convention as scripts/aggregate_all_v2.py (highest-step val row; mean±std on
the %-scale), groups by (wave, dataset, arch, variant) over seeds, and writes:

  tpami-bundle/added_exps/added_exps_per_cell.csv   per-cell mean/std/n/max_step
  tpami-bundle/added_exps/SUMMARY.md                per-wave markdown tables
  tpami-bundle/added_exps/README.md                 provenance + what each wave is

Idempotent and partial-completion safe: only cells with a final val row are
included; cells whose max step < 100000 are flagged in-progress. Re-run any
time; re-run once all 163 jobs finish for the final collection.

  python3 scripts/collect_added_exps.py
"""
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ELEC = Path("/scratch/elec/t41020_egovla")
V2_ROOT = ELEC / "slotcontrast_v2/v1_100k/slotcontrast_v2_100k"
PA_ROOT = ELEC / "slotcontrast_phase_a/v1_100k/slotcontrast_phase_a"
PB_ROOT = ELEC / "slotcontrast_phase_b/v1_100k/slotcontrast_phase_b"
OUT = Path(__file__).resolve().parent.parent / "tpami-bundle" / "added_exps"
OUT.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = ("val/ari", "val/fg_ari", "val/mbo", "val/image_ari", "val/image_mbo")
TS = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"

# wave -> (root, compiled regex with named groups dataset/arch/variant/seed)
WAVES = {
    "frozen_nomatch": (V2_ROOT, re.compile(
        TS + r"(?P<dataset>movic|movid|movie|ytvis)_(?P<arch>sc|gcv1_pf)_"
        r"(?P<variant>frozen_nomatch)_v2_s(?P<seed>\d+)$")),
    "lastk": (PB_ROOT, re.compile(
        TS + r"phaseB_(?P<dataset>movic|movid|movie|ytvis)_(?P<arch>sc|gcv1_pf)_"
        r"(?P<variant>lastk2|lastk4)_s(?P<seed>\d+)$")),
    "lora_lr": (PA_ROOT, re.compile(
        TS + r"phaseA_(?P<dataset>movic|movid|movie|ytvis)_(?P<arch>sc|gcv1_pf)_"
        r"lora_st_fr15_(?P<variant>lr[0-9eE.-]+)_s(?P<seed>\d+)$")),
    "mae_learned": (PA_ROOT, re.compile(
        TS + r"phaseA_(?P<dataset>ytvis)_(?P<arch>sc)_mae_native224_"
        r"(?P<variant>learned)_s(?P<seed>\d+)$")),
}

# expected cell counts per wave (for completion reporting)
EXPECTED = {"frozen_nomatch": 40, "lastk": 60, "lora_lr": 60, "mae_learned": 3}

# Only THIS campaign's runs: dir timestamp (YYYY-MM-DD-HH-MM-SS, lexicographically
# chronological) at/after the launch. Excludes pre-existing MOVi-D last-k / LoRA-LR
# cells (earlier phases) that share the reused phase_a/phase_b roots and naming.
LAUNCH_FLOOR = "2026-05-29-10-00-00"
# Lightning logs the last val at step 99999 (0-indexed) for a 100K run; use a
# tolerant floor so a finished cell counts as complete.
COMPLETE_STEP = 99000


def parse_final_val(csv_path: Path):
    """(highest_val_step, {metric: float}) or None. Matches aggregate_all_v2.py."""
    last_row, last_step = None, -1
    try:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if not row.get("val/ari", "").strip():
                    continue
                s = row.get("step", "").strip()
                step = int(s) if s else -1
                if step >= last_step:
                    last_step, last_row = step, row
    except (OSError, ValueError):
        return None
    if last_row is None:
        return None
    return last_step, {k: (float(last_row[k]) if last_row.get(k, "").strip()
                           else float("nan")) for k in METRIC_KEYS}


def fmt(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return "—"
    p = [v * 100 for v in vals]
    m = statistics.mean(p)
    s = statistics.stdev(p) if len(p) >= 2 else 0.0
    return f"{m:.2f} ± {s:.2f}" if s > 0 else f"{m:.2f}"


def collect():
    # (wave, dataset, arch, variant) -> seed -> (step, metrics)
    cells = defaultdict(dict)
    n_runs = 0
    for wave, (root, rgx) in WAVES.items():
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if d.name[:19] < LAUNCH_FLOOR:   # exclude pre-existing (earlier-dated) runs
                continue
            m = rgx.match(d.name)
            if not m:
                continue
            res = parse_final_val(d / "metrics" / "version_0" / "metrics.csv")
            if res is None:
                continue
            step, vals = res
            key = (wave, m.group("dataset"), m.group("arch"), m.group("variant"))
            seed = int(m.group("seed"))
            prev = cells[key].get(seed)
            if prev is None or step > prev[0]:
                cells[key][seed] = (step, vals)
                n_runs += 1
    return cells


def main():
    cells = collect()
    # per-cell CSV
    csv_path = OUT / "added_exps_per_cell.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wave", "dataset", "arch", "variant", "n_seeds",
                    "min_step", "all_100k",
                    "ari_mean", "ari_std", "fg_ari_mean", "fg_ari_std",
                    "mbo_mean", "mbo_std"])
        for (wave, ds, arch, var), seeds in sorted(cells.items()):
            steps = [s for s, _ in seeds.values()]
            def col(k):
                return [v[k] for _, v in seeds.values() if v[k] == v[k]]
            def ms(xs):
                xs = [x * 100 for x in xs]
                if not xs:
                    return ("", "")
                return (f"{statistics.mean(xs):.2f}",
                        f"{statistics.stdev(xs):.2f}" if len(xs) >= 2 else "0.00")
            am, asd = ms(col("val/ari"))
            fm, fsd = ms(col("val/fg_ari"))
            bm, bsd = ms(col("val/mbo"))
            w.writerow([wave, ds, arch, var, len(seeds), min(steps),
                        int(min(steps) >= COMPLETE_STEP), am, asd, fm, fsd, bm, bsd])

    # completion report + summary tables
    lines = ["# Added-experiments collection — breadth campaign (2026-05-29)\n"]
    lines.append("Final-validation metrics (highest-step val row per run, "
                 "mean ± std over seeds, %-scale), same convention as "
                 "the main v2 aggregator.\n")
    for wave in WAVES:
        wave_cells = {k: v for k, v in cells.items() if k[0] == wave}
        n_runs = sum(len(v) for v in wave_cells.values())
        complete = sum(1 for v in wave_cells.values()
                       for s, _ in v.values() if s >= COMPLETE_STEP)
        lines.append(f"\n## {wave}  ({n_runs}/{EXPECTED[wave]} runs found, "
                     f"{complete} at 100K)\n")
        lines.append("| dataset | arch | variant | n | min step | ARI | FG-ARI | mBO |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for (w, ds, arch, var), seeds in sorted(wave_cells.items()):
            steps = [s for s, _ in seeds.values()]
            def col(k):
                return [v[k] for _, v in seeds.values()]
            lines.append(
                f"| {ds} | {arch} | {var} | {len(seeds)} | {min(steps)} | "
                f"{fmt(col('val/ari'))} | {fmt(col('val/fg_ari'))} | "
                f"{fmt(col('val/mbo'))} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    total = sum(len(v) for v in cells.values())
    print(f"collected {total} runs across {len(cells)} cells")
    for wave in WAVES:
        n = sum(len(v) for k, v in cells.items() if k[0] == wave)
        c = sum(1 for k, v in cells.items() if k[0] == wave
                for s, _ in v.values() if s >= COMPLETE_STEP)
        print(f"  {wave:16s} {n:3d}/{EXPECTED[wave]} runs ({c} at 100K)")
    print(f"wrote {csv_path}")
    print(f"wrote {OUT/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
