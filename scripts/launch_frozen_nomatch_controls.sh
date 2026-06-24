#!/bin/bash
# Missing factorial control: FROZEN backbone × mechanism-OFF, at 100K, n=5.
#
# The v2 grid already has three corners of the {freeze, adapt} × {mechanism
# on, off} factorial:
#   - frozen            = frozen backbone + default predictor/matching   (mechanism ON)
#   - st/hung/default   = adapted (LoRA) backbone + predictor/matching    (mechanism ON)
#   - nomatch           = adapted (LoRA) backbone + skip_predictor=true   (mechanism OFF)
# The MISSING corner is frozen backbone + skip_predictor=true (mechanism OFF
# WITHOUT adaptation). These runs add it:
#   1. SC   frozen + identity propagation   -> "does SC need its learned
#                                               temporal predictor when NOT adapting?"
#   2. GCv1 frozen + no correspondence      -> "does GC need correspondence
#                                               under the conference/frozen setting?"
#
# Each new cell differs from the EXISTING `frozen` baseline by exactly one
# key (latent_processor.skip_predictor: true) — same frozen backbone, same
# per-dataset LR, same schedule — so frozen_nomatch vs frozen is a clean
# 1-variable comparison.
#
# Variant snippets (already exist, created during the oral push at 15K;
# build_100k_snippet.py auto-bumps the schedule to 100K):
#   SC   : configs/slotcontrast/snippets/sc_nomatch_frozen_15k.yaml
#   GCv1 : configs/slotcontrast/snippets/gcv1_nomatch_frozen_15k.yaml
#
# Save root + experiment_group reuse the v2 grid so scripts/aggregate_all_v2.py
# discovers these cells automatically (add `frozen_nomatch` to its
# CORE_VARIANTS list after the runs land to surface them in the grid MD).
#
# Matrix: 2 archs × 4 datasets × 5 seeds = 40 runs.
#
# Usage:
#   bash scripts/launch_frozen_nomatch_controls.sh                 # DRY-RUN (default)
#   bash scripts/launch_frozen_nomatch_controls.sh smoke --live    # 1k-step gpu-debug smoke (1 SC + 1 GCv1)
#   bash scripts/launch_frozen_nomatch_controls.sh runs  --live    # submit all 40 100K runs
# Nothing is submitted unless `--live` is passed.

set -euo pipefail

LIVE=0
MODE=""
for a in "$@"; do
  case "$a" in
    --live|-y|--yes) LIVE=1 ;;
    smoke|runs|all) MODE="$MODE $a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
MODE="${MODE:-help}"

TRITON_SLURM="triton_slurm.sh"
SNIP=configs/slotcontrast/snippets
SNIP_OUT_DIR="configs/slotcontrast/snippets/v2_generated"
mkdir -p "$SNIP_OUT_DIR"

set +u
module load mamba 2>/dev/null || true
source activate slotcontrast 2>/dev/null || conda activate slotcontrast 2>/dev/null || true
set -u

# Reuse the v2 grid save root + group so the existing aggregator finds these.
export OUTPUT_DIR="/scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k"
SLURM_LOG_DIR="$OUTPUT_DIR/slurm_logs"
mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"
EXP_GROUP="slotcontrast_v2_100k"

# Dataset base configs (identical set used by launch_full_100k_v2.sh).
GC_MC_CFG="configs/slotcontrast/app/movi_c_gcv1_rescue15k.yaml"
GC_MD_CFG="configs/slotcontrast/app/movi_d_gcv1_rescue15k.yaml"
GC_ME_CFG="configs/slotcontrast/app/movi_e_gcv1_rescue15k.yaml"
GC_YT_CFG="configs/slotcontrast/app/ytvis2021_gcv1_rescue15k.yaml"
SC_MC_CFG="configs/slotcontrast/app/movi_c_slotcontrast_rescue15k.yaml"
SC_MD_CFG="configs/slotcontrast/app/movi_d_slotcontrast_rescue15k.yaml"
SC_ME_CFG="configs/slotcontrast/app/movi_e_slotcontrast_rescue15k.yaml"
SC_YT_CFG="configs/slotcontrast/app/ytvis2021_slotcontrast_rescue15k.yaml"

# Frozen + mechanism-off variant snippets (15K source; builder bumps to 100K).
SN_FRNM_SC="$SNIP/sc_nomatch_frozen_15k.yaml"
SN_FRNM_GC="$SNIP/gcv1_nomatch_frozen_15k.yaml"

# Submission infra (matches v2 grid exactly).
PARTITION="gpu-h200-141g-ellis"
TIME_100K="1-12:00:00"     # 36h walltime; ~25h expected at 100K
ACCOUNT="ellis_users"
COMMON_OVR="checkpoint_every_n_steps=25000"   # 4 ckpts per run
SEEDS="0 1 2 3 4"

DRY=$([ "$LIVE" = "1" ] && echo "0" || echo "1")
[ "$DRY" = "1" ] && echo ">>> DRY-RUN MODE (no jobs submitted). Pass --live to submit. <<<"

is_mode() { for m in $MODE; do [ "$m" = "$1" ] || [ "$m" = "all" ] && return 0; done; return 1; }

build_snippet() {
  local BASE=$1 VAR=$2 NAME=$3
  local OUT="$SNIP_OUT_DIR/${NAME}.yaml"
  if [ ! -f "$OUT" ] || [ "$VAR" -nt "$OUT" ] || [ "$BASE" -nt "$OUT" ]; then
    python3 scripts/build_100k_snippet.py --base "$BASE" --variant "$VAR" --out "$OUT" >/dev/null
  fi
  echo "$OUT"
}

submit() {
  local NAME=$1 CFG=$2 VAR=$3 SEED=$4 EXTRA="${5:-}"
  local SNIPPET_KEY=$(echo "${NAME}" | sed 's/_s[0-9]*$//')
  local SNIPPET=$(build_snippet "$CFG" "$VAR" "$SNIPPET_KEY")
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (cfg=$(basename "$CFG") variant=$(basename "$VAR") seed=$SEED extra=$EXTRA)"
    return
  fi
  sbatch --job-name="$NAME" --account="$ACCOUNT" \
    --partition="$PARTITION" --time="$TIME_100K" \
    --requeue \
    --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
    --output="$SLURM_LOG_DIR/${NAME}_%j.out" \
    --error="$SLURM_LOG_DIR/${NAME}_%j.err" \
    "$TRITON_SLURM" "$CFG" \
    "experiment_group=$EXP_GROUP" "experiment_name=$NAME" \
    "seed=$SEED" $COMMON_OVR $EXTRA \
    "--config_overrides_file=$SNIPPET"
}

# ----- SMOKE: 1 SC + 1 GCv1 frozen+nomatch, 1k steps on gpu-debug -----
if is_mode "smoke"; then
  echo "=== SMOKE: frozen+nomatch wiring (1k steps, gpu-debug) ==="
  for spec in "movid_sc_frozen_nomatch:$SC_MD_CFG:$SN_FRNM_SC:" \
              "movid_gcv1_pf_frozen_nomatch:$GC_MD_CFG:$SN_FRNM_GC:model.initializer.init_mode=per_frame"; do
    IFS=':' read -r BNAME CFG VAR EXTRA <<< "$spec"
    NAME="smoke_${BNAME}_1k"
    SNIPPET=$(build_snippet "$CFG" "$VAR" "$NAME")
    if [ "$DRY" = "1" ]; then
      echo "DRY: $NAME (cfg=$(basename "$CFG") variant=$(basename "$VAR") extra=$EXTRA) snippet=$SNIPPET"
    else
      sbatch --job-name="$NAME" --account="aalto_users" \
        --partition="gpu-debug" --time="0-00:30:00" --requeue \
        --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
        --output="$SLURM_LOG_DIR/${NAME}_%j.out" \
        --error="$SLURM_LOG_DIR/${NAME}_%j.err" \
        "$TRITON_SLURM" "$CFG" \
        "experiment_group=${EXP_GROUP}_smoke" "experiment_name=$NAME" \
        "seed=0" "trainer.max_steps=1000" "trainer.val_check_interval=500" \
        "checkpoint_every_n_steps=500" $EXTRA \
        "--config_overrides_file=$SNIPPET"
    fi
  done
fi

# ----- RUNS: 40 frozen+nomatch controls @ 100K, n=5 -----
if is_mode "runs"; then
  echo "=== frozen+nomatch controls @ 100K, n=5 — 40 runs ==="
  for SEED in $SEEDS; do
    # SlotContrast frozen + identity propagation (no init_mode override)
    submit "movic_sc_frozen_nomatch_v2_s${SEED}" "$SC_MC_CFG" "$SN_FRNM_SC" "$SEED"
    submit "movid_sc_frozen_nomatch_v2_s${SEED}" "$SC_MD_CFG" "$SN_FRNM_SC" "$SEED"
    submit "movie_sc_frozen_nomatch_v2_s${SEED}" "$SC_ME_CFG" "$SN_FRNM_SC" "$SEED"
    submit "ytvis_sc_frozen_nomatch_v2_s${SEED}" "$SC_YT_CFG" "$SN_FRNM_SC" "$SEED"
    # Grounded Correspondence frozen + no correspondence (init_mode=per_frame)
    submit "movic_gcv1_pf_frozen_nomatch_v2_s${SEED}" "$GC_MC_CFG" "$SN_FRNM_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_frozen_nomatch_v2_s${SEED}" "$GC_MD_CFG" "$SN_FRNM_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_frozen_nomatch_v2_s${SEED}" "$GC_ME_CFG" "$SN_FRNM_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_frozen_nomatch_v2_s${SEED}" "$GC_YT_CFG" "$SN_FRNM_GC" "$SEED" "model.initializer.init_mode=per_frame"
  done
fi

if [ "$MODE" = "help" ] || { ! is_mode "smoke" && ! is_mode "runs"; }; then
  echo "Usage: bash $(basename "$0") {smoke|runs|all} [--live]"
  echo "  smoke: 1 SC + 1 GCv1 frozen+nomatch, 1k steps on gpu-debug (~5 min)"
  echo "  runs : 40 100K runs (2 archs × 4 datasets × 5 seeds)"
  echo "Without --live the script is DRY and calls no sbatch."
  echo "Save root: $OUTPUT_DIR  | experiment_group: $EXP_GROUP"
fi
