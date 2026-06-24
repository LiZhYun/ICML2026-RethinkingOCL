"""
Paired-bootstrap CIs + Holm-Bonferroni-corrected p-values for the headline
comparisons in EXPERIMENTAL_RESULTS.md.

For each (method_A, method_B, dataset, metric) we have n=5 seeds (or n=3 for
the cross-backbone grid). We compute:

  - Per-seed paired difference d_s = A_s - B_s.
  - Mean difference, percentile bootstrap 95% CI (B=10_000).
  - Paired t-test p-value (two-sided; the only test with non-trivial power
    at n=5 -- exact sign test caps at p=2/32=0.0625 and exact paired
    permutation caps at the same value).
  - Holm-Bonferroni-adjusted p-value within each pre-registered family.

Three pre-registered families:
  F1 ("rescue lift"):       SC st     vs SC frozen, 4 datasets x 3 metrics = 12 tests
  F2 ("predictor ablation"): SC st     vs SC nomatch, 4 datasets x 3 metrics = 12 tests
  F3 ("cross-backbone identity vs learned"): 3 backbones x 3 metrics = 9 tests

Random seed is fixed so the bootstrap is reproducible.
"""

from __future__ import annotations
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RNG = np.random.default_rng(0)
B = 10_000  # bootstrap replicates

V2_ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k/slotcontrast_v2_100k"
)
XBB_ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_cross_backbone/v1_100k/"
    "slotcontrast_cross_backbone"
)
PHASE_A_ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_phase_a/v1_100k/slotcontrast_phase_a"
)
PHASE_B_ROOT = Path(
    "/scratch/elec/t41020_egovla/slotcontrast_phase_b/v1_100k/slotcontrast_phase_b"
)
OUT_PATH = Path("/scratch/elec/t41020_egovla/paired_stats.md")

V2_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"(?P<dataset>movic|movid|movie|ytvis)_"
    r"(?P<arch>sc|gcv1_pf)_"
    r"(?P<variant>[a-zA-Z0-9_]+?)"
    r"_v2_s(?P<seed>\d+)$"
)
XBB_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"ytvis_sc_(?P<bb>dinov[123])_(?P<variant>learned|identity)_s(?P<seed>\d+)$"
)

METRIC_KEYS = ("val/ari", "val/fg_ari", "val/mbo")


def parse_final_val(csv_path: Path) -> dict[str, float] | None:
    last_val = None
    last_step = -1
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                ari = row.get("val/ari", "").strip()
                if not ari:
                    continue
                step = int(row.get("step") or -1)
                if step >= last_step:
                    last_step = step
                    last_val = row
    except (OSError, ValueError):
        return None
    if last_val is None:
        return None
    out = {}
    for k in METRIC_KEYS:
        v = last_val.get(k, "").strip()
        out[k] = float(v) if v else math.nan
    return out


def parse_final_val_with_step(csv_path: Path) -> tuple[int, dict[str, float]] | None:
    """Same as parse_final_val, but also returns the val step it came from."""
    last_val = None
    last_step = -1
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                ari = row.get("val/ari", "").strip()
                if not ari:
                    continue
                step = int(row.get("step") or -1)
                if step >= last_step:
                    last_step = step
                    last_val = row
    except (OSError, ValueError):
        return None
    if last_val is None:
        return None
    out = {}
    for k in METRIC_KEYS:
        v = last_val.get(k, "").strip()
        out[k] = float(v) if v else math.nan
    return last_step, out


def collect_v2() -> dict[tuple, dict[int, dict[str, float]]]:
    # (dataset, arch, variant) -> seed -> metrics; dedup by HIGHEST step.
    best: dict[tuple, dict[int, tuple[int, dict[str, float]]]] = defaultdict(dict)
    for d in sorted(V2_ROOT.iterdir()):
        m = V2_RE.match(d.name)
        if not m:
            continue
        cell = (m.group("dataset"), m.group("arch"), m.group("variant"))
        seed = int(m.group("seed"))
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val_with_step(csv_path)
        if res is None:
            continue
        step, vals = res
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)
    return {cell: {s: v for s, (_, v) in seeds.items()} for cell, seeds in best.items()}


PHASE_A_T12_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"phaseA_(?P<dataset>movic|movid|movie|ytvis)_(?P<arch>sc|gcv1_pf)_fullft_lr(?P<lr>1e-5|1e-4|4e-4)_s(?P<seed>\d+)$"
)
PHASE_B_BITFIT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"phaseB_(?P<dataset>movic|movid|movie|ytvis)_(?P<arch>sc|gcv1_pf)_bitfit_s(?P<seed>\d+)$"
)
PHASE_B_LASTK_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_"
    r"phaseB_movid_(?P<arch>sc|gcv1_pf)_lastk(?P<k>2|4)_s(?P<seed>\d+)$"
)


def collect_phase_a_fullft_lr() -> dict[tuple, dict[int, dict[str, float]]]:
    """(dataset, arch, lr) -> seed -> metrics. Phase A T1.2 + T1.2-extra fullft LR sweep
    on MOVi-D (3 LRs) and MOVi-C/E (lr=1e-5 only, from F.4 grid)."""
    best: dict[tuple, dict[int, tuple[int, dict[str, float]]]] = defaultdict(dict)
    if not PHASE_A_ROOT.exists():
        return {}
    for d in sorted(PHASE_A_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = PHASE_A_T12_RE.match(d.name)
        if not m:
            continue
        cell = (m.group("dataset"), m.group("arch"), m.group("lr"))
        seed = int(m.group("seed"))
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val_with_step(csv_path)
        if res is None:
            continue
        step, vals = res
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)
    return {cell: {s: v for s, (_, v) in seeds.items()} for cell, seeds in best.items()}


def collect_phase_b_bitfit() -> dict[tuple, dict[int, dict[str, float]]]:
    """(dataset, arch) -> seed -> metrics. Phase B T3.1 BitFit."""
    best: dict[tuple, dict[int, tuple[int, dict[str, float]]]] = defaultdict(dict)
    if not PHASE_B_ROOT.exists():
        return {}
    for d in sorted(PHASE_B_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = PHASE_B_BITFIT_RE.match(d.name)
        if not m:
            continue
        cell = (m.group("dataset"), m.group("arch"))
        seed = int(m.group("seed"))
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val_with_step(csv_path)
        if res is None:
            continue
        step, vals = res
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)
    return {cell: {s: v for s, (_, v) in seeds.items()} for cell, seeds in best.items()}


def collect_phase_b_lastk() -> dict[tuple, dict[int, dict[str, float]]]:
    """(arch, k) -> seed -> metrics. Phase B T3.2 last-k partial unfreeze on MOVi-D."""
    best: dict[tuple, dict[int, tuple[int, dict[str, float]]]] = defaultdict(dict)
    if not PHASE_B_ROOT.exists():
        return {}
    for d in sorted(PHASE_B_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = PHASE_B_LASTK_RE.match(d.name)
        if not m:
            continue
        cell = (m.group("arch"), m.group("k"))
        seed = int(m.group("seed"))
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val_with_step(csv_path)
        if res is None:
            continue
        step, vals = res
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)
    return {cell: {s: v for s, (_, v) in seeds.items()} for cell, seeds in best.items()}


def collect_xbb() -> dict[tuple, dict[int, dict[str, float]]]:
    # Same highest-step dedup as collect_v2.
    best: dict[tuple, dict[int, tuple[int, dict[str, float]]]] = defaultdict(dict)
    for d in sorted(XBB_ROOT.iterdir()):
        m = XBB_RE.match(d.name)
        if not m:
            continue
        cell = (m.group("bb"), m.group("variant"))
        seed = int(m.group("seed"))
        csv_path = d / "metrics" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        res = parse_final_val_with_step(csv_path)
        if res is None:
            continue
        step, vals = res
        prev = best[cell].get(seed)
        if prev is None or step > prev[0]:
            best[cell][seed] = (step, vals)
    return {cell: {s: v for s, (_, v) in seeds.items()} for cell, seeds in best.items()}


def paired_diff(
    cell_A: dict[int, dict[str, float]],
    cell_B: dict[int, dict[str, float]],
    metric: str,
) -> np.ndarray:
    """Return d_s = A_s - B_s on the %-scale for the seeds present in BOTH cells."""
    seeds = sorted(set(cell_A) & set(cell_B))
    diffs = []
    for s in seeds:
        a = cell_A[s][metric]
        b = cell_B[s][metric]
        if math.isnan(a) or math.isnan(b):
            continue
        diffs.append((a - b) * 100.0)
    return np.array(diffs, dtype=float)


def bootstrap_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean_diff, lower, upper) via paired percentile bootstrap."""
    n = len(diffs)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    boot_means = np.empty(B)
    for i in range(B):
        idx = RNG.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    mean = float(diffs.mean())
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return mean, lo, hi


def paired_ttest_p(diffs: np.ndarray) -> float:
    """Two-sided paired t-test on the differences (one-sample t-test on d)."""
    if len(diffs) < 2:
        return math.nan
    t, p = stats.ttest_1samp(diffs, popmean=0.0, alternative="two-sided")
    if math.isnan(p):
        return math.nan
    return float(p)


def tost_p(diffs: np.ndarray, delta: float) -> float:
    """Two One-Sided Tests (Schuirmann) for paired equivalence within ±delta.

    Returns the TOST p-value = max(p_lower, p_upper), where
      p_lower = P(t <= t_lower)  with t_lower = (mean(d) - (-delta)) / SE
                                 testing H0a: mean(d) <= -delta
      p_upper = P(t >= t_upper)  with t_upper = (mean(d) - (+delta)) / SE
                                 testing H0b: mean(d) >= +delta
    If TOST p < alpha, conclude equivalence within [-delta, +delta].
    """
    n = len(diffs)
    if n < 2 or delta <= 0:
        return math.nan
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n))
    if se == 0:
        # Zero variance: deterministic difference. Equivalence iff |mean| < delta.
        return 0.0 if abs(mean) < delta else 1.0
    df = n - 1
    t_lower = (mean - (-delta)) / se
    t_upper = (mean - (+delta)) / se
    p_lower = float(stats.t.sf(t_lower, df))   # P(T > t_lower) under H0a
    p_upper = float(stats.t.cdf(t_upper, df))  # P(T < t_upper) under H0b
    return max(p_lower, p_upper)


def holm_bonferroni(ps: list[float]) -> list[float]:
    """Return Holm-Bonferroni-adjusted p-values in original order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj_sorted = []
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = ps[idx]
        if math.isnan(raw):
            adj_sorted.append(math.nan)
            continue
        adj = raw * (m - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adj_sorted.append(running_max)
    out = [0.0] * m
    for sorted_pos, original_idx in enumerate(order):
        out[original_idx] = adj_sorted[sorted_pos]
    return out


# ----------------------------------------------------------------------
def run_family(
    name: str,
    comps: list[tuple[str, dict, dict, str]],  # (label, cell_A, cell_B, metric)
    out_lines: list[str],
) -> None:
    rows = []
    raw_ps = []
    for label, A, B, metric in comps:
        diffs = paired_diff(A, B, metric)
        n = len(diffs)
        if n < 2:
            rows.append((label, n, math.nan, math.nan, math.nan, math.nan))
            raw_ps.append(math.nan)
            continue
        mean, lo, hi = bootstrap_ci(diffs)
        p = paired_ttest_p(diffs)
        rows.append((label, n, mean, lo, hi, p))
        raw_ps.append(p)

    adj_ps = holm_bonferroni(raw_ps)

    out_lines.append(f"\n### {name}\n")
    out_lines.append("| Comparison | n | Δ mean | 95% CI | raw p | Holm p | reject @0.05? |")
    out_lines.append("|---|---:|---:|---|---:|---:|:---:|")
    for (label, n, mean, lo, hi, p), pa in zip(rows, adj_ps):
        if n < 2 or math.isnan(p):
            out_lines.append(f"| {label} | {n} | — | — | — | — | — |")
            continue
        ci = f"[{lo:+.2f}, {hi:+.2f}]"
        ci_excl_zero = "**yes**" if (lo > 0 or hi < 0) else "no"
        rej = "**yes**" if pa < 0.05 else "no"
        out_lines.append(
            f"| {label} | {n} | {mean:+.2f} | {ci} | {p:.4g} | {pa:.4g} | {rej} |"
        )
    out_lines.append("")


def run_tost_family(
    name: str,
    comps: list[tuple[str, dict, dict, str]],
    margins: dict[str, float],
    out_lines: list[str],
) -> None:
    """TOST equivalence test with per-metric pre-registered margins.

    margins: dict mapping metric -> ±delta (e.g. {"val/ari": 2.0, "val/mbo": 1.0}).
    Holm-Bonferroni applied to the TOST p-values within this family.
    """
    rows = []
    raw_ps = []
    for label, A, B, metric in comps:
        diffs = paired_diff(A, B, metric)
        n = len(diffs)
        delta = margins.get(metric, math.nan)
        if n < 2 or math.isnan(delta):
            rows.append((label, n, math.nan, math.nan, math.nan, delta, math.nan))
            raw_ps.append(math.nan)
            continue
        mean, lo, hi = bootstrap_ci(diffs)
        p = tost_p(diffs, delta)
        rows.append((label, n, mean, lo, hi, delta, p))
        raw_ps.append(p)
    adj_ps = holm_bonferroni(raw_ps)

    out_lines.append(f"\n### {name}\n")
    out_lines.append(
        "| Comparison | n | Δ mean | 95% CI | margin ±δ | TOST p | Holm p | equivalent @0.05? |"
    )
    out_lines.append("|---|---:|---:|---|---:|---:|---:|:---:|")
    for (label, n, mean, lo, hi, delta, p), pa in zip(rows, adj_ps):
        if n < 2 or math.isnan(p):
            out_lines.append(f"| {label} | {n} | — | — | — | — | — | — |")
            continue
        ci = f"[{lo:+.2f}, {hi:+.2f}]"
        eq = "**yes**" if pa < 0.05 else "no"
        out_lines.append(
            f"| {label} | {n} | {mean:+.2f} | {ci} | ±{delta:.1f} | {p:.4g} | {pa:.4g} | {eq} |"
        )
    out_lines.append("")


def main() -> None:
    print("Collecting v2 grid...")
    v2 = collect_v2()
    print(f"  cells={len(v2)}")
    print("Collecting cross-backbone...")
    xbb = collect_xbb()
    print(f"  cells={len(xbb)}")
    print("Collecting Phase A T1.2 fullft LR sweep...")
    pa_lr = collect_phase_a_fullft_lr()
    print(f"  cells={len(pa_lr)}")
    print("Collecting Phase B T3.1 BitFit...")
    pb_bitfit = collect_phase_b_bitfit()
    print(f"  cells={len(pb_bitfit)}")
    print("Collecting Phase B T3.2 last-k...")
    pb_lastk = collect_phase_b_lastk()
    print(f"  cells={len(pb_lastk)}")

    out: list[str] = []
    out.append("# Paired-bootstrap CIs and Holm-Bonferroni p-values\n")
    out.append(f"Bootstrap replicates: B={B:,}. Paired t-test, two-sided. "
               f"Holm correction is per pre-registered family below.\n")

    # --- F1: rescue lift on SC ---
    F1 = []
    for ds in ("movic", "movid", "movie", "ytvis"):
        ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                   "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
        A = v2[(ds, "sc", "st")]
        B0 = v2[(ds, "sc", "frozen")]
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F1.append((f"SC {ds_name} {mname}: st - frozen", A, B0, metric))
    run_family("F1 — SC rescue lift (st − frozen), 12 tests", F1, out)

    # --- F2: predictor ablation on SC (st vs nomatch) ---
    F2 = []
    for ds in ("movic", "movid", "movie", "ytvis"):
        ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                   "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
        A = v2[(ds, "sc", "st")]
        B0 = v2[(ds, "sc", "nomatch")]
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F2.append((f"SC {ds_name} {mname}: st - nomatch", A, B0, metric))
    run_family("F2 — SC predictor ablation (st − nomatch), 12 tests", F2, out)

    # --- F2a-c: other learned/structured predictors vs nomatch on SC ---
    for variant_key, variant_label, family_letter in (
        ("default", "default (TransformerEncoder)", "a"),
        ("hung", "Hungarian", "b"),
        ("noST", "Soft-SoftIdent (no-ST)", "c"),
    ):
        comps = []
        for ds in ("movic", "movid", "movie", "ytvis"):
            ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                       "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
            cell = v2.get((ds, "sc", variant_key), {})
            B0 = v2[(ds, "sc", "nomatch")]
            if not cell:
                continue
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                comps.append(
                    (f"SC {ds_name} {mname}: {variant_key} - nomatch",
                     cell, B0, metric)
                )
        if comps:
            run_family(
                f"F2{family_letter} — SC predictor ablation "
                f"({variant_label} − nomatch), {len(comps)} tests",
                comps, out,
            )

    # --- F3: GCv1 rescue lift ---
    F3 = []
    for ds in ("movic", "movid", "movie", "ytvis"):
        ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                   "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
        A = v2[(ds, "gcv1_pf", "st")]
        B0 = v2[(ds, "gcv1_pf", "frozen")]
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F3.append((f"GCv1 {ds_name} {mname}: st - frozen", A, B0, metric))
    run_family("F3 — GCv1 rescue lift (st − frozen), 12 tests", F3, out)

    # --- F4: GCv1 frozen − nomatch (predictor MATTERS for GCv1) ---
    F4 = []
    for ds in ("movic", "movid", "movie", "ytvis"):
        ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                   "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
        A = v2[(ds, "gcv1_pf", "frozen")]
        B0 = v2[(ds, "gcv1_pf", "nomatch")]
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F4.append((f"GCv1 {ds_name} {mname}: frozen - nomatch", A, B0, metric))
    run_family("F4 — GCv1 predictor necessity (frozen − nomatch), 12 tests", F4, out)

    # --- F4c: GCv1 CLEAN single-variable predictor necessity ---
    # F4 above uses the LoRA-ADAPTED nomatch cell, so it confounds "remove the
    # predictor" with "adapt the backbone". F4c holds the backbone FROZEN and
    # toggles only skip_predictor (frozen vs frozen_nomatch), so it is the pure
    # single-variable contrast. frozen_nomatch cells are auto-discovered by
    # collect_v2() (same v2 root, _v2_s<seed> suffix), n=5.
    F4c = []
    for ds in ("movic", "movid", "movie", "ytvis"):
        ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                   "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
        A = v2.get((ds, "gcv1_pf", "frozen"), {})
        B0 = v2.get((ds, "gcv1_pf", "frozen_nomatch"), {})
        if not A or not B0:
            continue
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F4c.append((f"GCv1 {ds_name} {mname}: frozen - frozen_nomatch", A, B0, metric))
    if F4c:
        run_family("F4c — GCv1 predictor necessity, CLEAN single-variable "
                   "(frozen − frozen+identity), 12 tests", F4c, out)

    # --- F2c: SC CLEAN single-variable predictor (non-)necessity ---
    # Companion to F2 (which used the adapted st vs adapted nomatch). F2c holds
    # the backbone FROZEN and toggles only skip_predictor — does SC need a learned
    # predictor even without adaptation? Expect ~0 (identity propagation suffices).
    F2c = []
    for ds in ("movic", "movid", "movie", "ytvis"):
        ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                   "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
        A = v2.get((ds, "sc", "frozen"), {})
        B0 = v2.get((ds, "sc", "frozen_nomatch"), {})
        if not A or not B0:
            continue
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F2c.append((f"SC {ds_name} {mname}: frozen - frozen_nomatch", A, B0, metric))
    if F2c:
        run_family("F4d — SlotContrast predictor (non-)necessity, CLEAN single-variable "
                   "(frozen − frozen+identity), 12 tests", F2c, out)

    # --- F5: cross-backbone identity vs learned ---
    F5 = []
    for bb in ("dinov1", "dinov2", "dinov3"):
        bb_name = {"dinov1": "DINOv1", "dinov2": "DINOv2", "dinov3": "DINOv3"}[bb]
        A = xbb[(bb, "identity")]
        B0 = xbb[(bb, "learned")]
        for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
            F5.append((f"{bb_name} {mname}: identity - learned", A, B0, metric))
    run_family(
        "F5 — Cross-backbone identity vs learned (n=3 each), 9 tests",
        F5, out,
    )

    # --- F2-TOST: equivalence test for the predictor ablation ---
    # Pre-registered margins (recorded here as a stable per-metric mapping):
    #   ARI / FG-ARI: ±2 percentage points
    #   mBO:          ±1 percentage point
    # These are tighter than the per-cell std on most rows, so they are
    # genuinely informative bounds for the "identity ≈ learned" claim.
    EQ_MARGINS = {"val/ari": 2.0, "val/fg_ari": 2.0, "val/mbo": 1.0}
    out.append(
        "\n---\n"
        "\n## Equivalence tests (TOST, Schuirmann)\n"
        "\nPre-registered margins: ±2 percentage points for ARI / FG-ARI, "
        "±1 percentage point for mBO. TOST p < 0.05 ⇒ reject the union "
        "$\\\\{\\Delta \\le -\\delta\\}\\cup\\{\\Delta \\ge +\\delta\\}$, "
        "i.e. conclude paired-mean equivalence within the pre-registered "
        "interval $[-\\delta, +\\delta]$. Holm correction applied within "
        "each TOST family below.\n"
    )
    run_tost_family(
        "F2-TOST — SC predictor ablation (st ≡ nomatch?), 12 tests",
        F2, EQ_MARGINS, out,
    )
    # TOST for other-predictor families
    for variant_key, variant_label, family_letter in (
        ("default", "default (TransformerEncoder)", "a"),
        ("hung", "Hungarian", "b"),
        ("noST", "Soft-SoftIdent (no-ST)", "c"),
    ):
        comps = []
        for ds in ("movic", "movid", "movie", "ytvis"):
            ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                       "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
            cell = v2.get((ds, "sc", variant_key), {})
            B0 = v2[(ds, "sc", "nomatch")]
            if not cell:
                continue
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                comps.append(
                    (f"SC {ds_name} {mname}: {variant_key} - nomatch",
                     cell, B0, metric)
                )
        if comps:
            run_tost_family(
                f"F2{family_letter}-TOST — SC ({variant_label} ≡ nomatch?), "
                f"{len(comps)} tests",
                comps, EQ_MARGINS, out,
            )
    run_tost_family(
        "F5-TOST — Cross-backbone identity ≡ learned?, 9 tests",
        F5, EQ_MARGINS, out,
    )

    # --- F6: SC/GCv1 LoRA-rescue vs fullft@1e-5 across MOVi-C/D/E + YT-VIS ---
    # Phase A T1.2 (MOVi-D) + F.4 (MOVi-C, MOVi-E) + Round-33 FIX-10 YT-VIS
    # extension. All at fullft encoder lr=1e-5 vs canonical LoRA-rescue.
    F6 = []
    for arch, arch_name in (("sc", "SC"), ("gcv1_pf", "GCv1")):
        for ds in ("movic", "movid", "movie", "ytvis"):
            ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                       "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
            A = v2.get((ds, arch, "st"), {})
            B0 = pa_lr.get((ds, arch, "1e-5"), {})
            if not A or not B0:
                continue
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                F6.append((f"{arch_name} {ds_name} {mname}: LoRA-rescue - fullft@1e-5",
                           A, B0, metric))
    run_family(
        "F6 — LoRA-rescue vs tuned full fine-tune (st − fullft@1e-5), "
        "up to 24 tests on MOVi-{C,D,E} + YT-VIS × {SC, GCv1} × 3 metrics. "
        "Note: MOVi-{ARI ≡ FG-ARI} by configuration (§2.1), so the "
        "inferentially unique-test count is 18 = 6 (MOVi cells × 2 unique "
        "metrics) + 6 (YT-VIS cells × 3 metrics) when collapsed.", F6, out,
    )

    # --- F7: BitFit vs frozen (Phase B T3.1) ---
    F7 = []
    for arch, arch_name in (("sc", "SC"), ("gcv1_pf", "GCv1")):
        for ds in ("movic", "movid", "movie", "ytvis"):
            ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                       "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
            A = pb_bitfit.get((ds, arch), {})
            B0 = v2.get((ds, arch, "frozen"), {})
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                F7.append((f"{arch_name} {ds_name} {mname}: BitFit - frozen",
                           A, B0, metric))
    run_family(
        "F7 — BitFit vs frozen (Phase B T3.1), 24 tests "
        "(4 datasets × 2 archs × 3 metrics)", F7, out,
    )

    # --- F8: BitFit vs LoRA-rescue ---
    F8 = []
    for arch, arch_name in (("sc", "SC"), ("gcv1_pf", "GCv1")):
        for ds in ("movic", "movid", "movie", "ytvis"):
            ds_name = {"movic": "MOVi-C", "movid": "MOVi-D",
                       "movie": "MOVi-E", "ytvis": "YT-VIS"}[ds]
            A = pb_bitfit.get((ds, arch), {})
            B0 = v2.get((ds, arch, "st"), {})
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                F8.append((f"{arch_name} {ds_name} {mname}: BitFit - LoRA-rescue",
                           A, B0, metric))
    run_family(
        "F8 — BitFit vs LoRA-rescue (Phase B T3.1 vs v2 st), 24 tests",
        F8, out,
    )

    # --- F9: last-k vs frozen + last-k vs LoRA-rescue (Phase B T3.2) ---
    F9 = []
    for arch, arch_name in (("sc", "SC"), ("gcv1_pf", "GCv1")):
        for k in ("2", "4"):
            A = pb_lastk.get((arch, k), {})
            B0_frozen = v2.get(("movid", arch, "frozen"), {})
            B0_rescue = v2.get(("movid", arch, "st"), {})
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                F9.append((f"{arch_name} MOVi-D {mname}: last-k{k} - frozen",
                           A, B0_frozen, metric))
            for metric, mname in zip(METRIC_KEYS, ("ARI", "FG-ARI", "mBO")):
                F9.append((f"{arch_name} MOVi-D {mname}: last-k{k} - LoRA-rescue",
                           A, B0_rescue, metric))
    run_family(
        "F9 — Last-k partial unfreeze vs frozen + LoRA-rescue (Phase B T3.2), "
        "24 tests (2 archs × 2 k × 2 baselines × 3 metrics)", F9, out,
    )

    OUT_PATH.write_text("\n".join(out))
    print(f"\nWrote: {OUT_PATH}")
    print("\n--- Preview ---")
    print("\n".join(out[:80]))


if __name__ == "__main__":
    main()
