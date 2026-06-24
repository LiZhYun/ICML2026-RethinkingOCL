#!/bin/bash
# Phase-B grid (Tier 3 — parameter-efficient adaptation alternatives).
#
# Cells:
#   T3.1  BitFit (bias-only fine-tuning):
#           {movic, movid, movie, ytvis} × {sc, gcv1_pf} × n=5 = 40 runs
#   T3.2  Last-k blocks partial-unfreeze:
#           movid × {sc, gcv1_pf} × {k=2, k=4} × n=5             = 20 runs
#
# Total: 60 100K-step runs.
# Save root: /scratch/elec/t41020_egovla/slotcontrast_phase_b/v1_100k/
# experiment_group: slotcontrast_phase_b
#
# Usage:
#   bash scripts/launch_phase_b.sh smoke --live   # 1k-step bitfit on gpu-debug
#   bash scripts/launch_phase_b.sh all   --live   # all 60 runs
#   bash scripts/launch_phase_b.sh all             # DRY-RUN (default)

set -euo pipefail

LIVE=0
WAVES=""
for a in "$@"; do
  case "$a" in
    --live|-y|--yes) LIVE=1 ;;
    smoke|t31|t32|all) WAVES="$WAVES $a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
WAVES="${WAVES:-help}"

TRITON_SLURM="triton_slurm.sh"
SNIP_PB="configs/slotcontrast/snippets/phase_b"

set +u
module load mamba 2>/dev/null || true
source activate slotcontrast 2>/dev/null || conda activate slotcontrast 2>/dev/null || true
set -u

export OUTPUT_DIR="/scratch/elec/t41020_egovla/slotcontrast_phase_b/v1_100k"
SLURM_LOG_DIR="$OUTPUT_DIR/slurm_logs"
mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"

EXP_GROUP="slotcontrast_phase_b"
PARTITION="gpu-h200-141g-ellis"
TIME_100K="1-12:00:00"
ACCOUNT="ellis_users"
COMMON_OVR="checkpoint_every_n_steps=25000"

# Base configs per (dataset, arch) — same as v2 launcher.
SC_MOVIC_CFG="configs/slotcontrast/app/movi_c_slotcontrast_rescue15k.yaml"
SC_MOVID_CFG="configs/slotcontrast/app/movi_d_slotcontrast_rescue15k.yaml"
SC_MOVIE_CFG="configs/slotcontrast/app/movi_e_slotcontrast_rescue15k.yaml"
SC_YT_CFG="configs/slotcontrast/app/ytvis2021_slotcontrast_rescue15k.yaml"
GC_MOVIC_CFG="configs/slotcontrast/app/movi_c_gcv1_rescue15k.yaml"
GC_MOVID_CFG="configs/slotcontrast/app/movi_d_gcv1_rescue15k.yaml"
GC_MOVIE_CFG="configs/slotcontrast/app/movi_e_gcv1_rescue15k.yaml"
GC_YT_CFG="configs/slotcontrast/app/ytvis2021_gcv1_rescue15k.yaml"

is_wave() { for w in $WAVES; do if [ "$w" = "$1" ] || [ "$w" = "all" ]; then return 0; fi; done; return 1; }
DRY=$([ "$LIVE" = "1" ] && echo "0" || echo "1")
if [ "$DRY" = "1" ]; then
  echo ">>> DRY-RUN MODE (no jobs will be submitted). Pass --live to submit. <<<"
fi

# submit name cfg snippet_path seed extra_dotlist
submit() {
  local NAME=$1 CFG=$2 SNIPPET=$3 SEED=$4 EXTRA="${5:-}"
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (cfg=$(basename "$CFG") snippet=$(basename "$SNIPPET") seed=$SEED extra=$EXTRA)"
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

# ===== SMOKE: BitFit (cheap — biases only) on gpu-debug =====
if is_wave "smoke"; then
  echo "=== SMOKE: 500-step BitFit on MOVi-C × SC × gpu-debug ==="
  if [ "$DRY" = "1" ]; then
    echo "DRY: smoke_movic_sc_bitfit_500"
  else
    sbatch --job-name="smoke_phaseB_bitfit_500" --account="aalto_users" \
      --partition="gpu-debug" --time="0-00:25:00" \
      --requeue \
      --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
      --output="$SLURM_LOG_DIR/smoke_phaseB_bitfit_500_%j.out" \
      --error="$SLURM_LOG_DIR/smoke_phaseB_bitfit_500_%j.err" \
      "$TRITON_SLURM" "$SC_MOVIC_CFG" \
      "experiment_group=${EXP_GROUP}_smoke" \
      "experiment_name=smoke_movic_sc_bitfit_500" \
      "seed=0" "trainer.max_steps=500" "trainer.val_check_interval=250" \
      "checkpoint_every_n_steps=250" \
      "--config_overrides_file=$SNIP_PB/bitfit_st_fr15.yaml"
  fi
fi

# ===== T3.1 — BitFit × 4 datasets × 2 archs =====
if is_wave "t31"; then
  echo "=== T3.1: BitFit × 4 datasets × 2 archs × n=5 ==="
  for SEED in 0 1 2 3 4; do
    submit "phaseB_movic_sc_bitfit_s${SEED}"      "$SC_MOVIC_CFG" "$SNIP_PB/bitfit_st_fr15.yaml" "$SEED"
    submit "phaseB_movid_sc_bitfit_s${SEED}"      "$SC_MOVID_CFG" "$SNIP_PB/bitfit_st_fr15.yaml" "$SEED"
    submit "phaseB_movie_sc_bitfit_s${SEED}"      "$SC_MOVIE_CFG" "$SNIP_PB/bitfit_st_fr15.yaml" "$SEED"
    submit "phaseB_ytvis_sc_bitfit_s${SEED}"      "$SC_YT_CFG"    "$SNIP_PB/bitfit_st_fr15_ytvis.yaml" "$SEED"
    # GCv1 (per_frame init via dotlist override, same as v2 launcher)
    submit "phaseB_movic_gcv1_pf_bitfit_s${SEED}" "$GC_MOVIC_CFG" "$SNIP_PB/bitfit_st_fr15.yaml" "$SEED" "model.initializer.init_mode=per_frame"
    submit "phaseB_movid_gcv1_pf_bitfit_s${SEED}" "$GC_MOVID_CFG" "$SNIP_PB/bitfit_st_fr15.yaml" "$SEED" "model.initializer.init_mode=per_frame"
    submit "phaseB_movie_gcv1_pf_bitfit_s${SEED}" "$GC_MOVIE_CFG" "$SNIP_PB/bitfit_st_fr15.yaml" "$SEED" "model.initializer.init_mode=per_frame"
    submit "phaseB_ytvis_gcv1_pf_bitfit_s${SEED}" "$GC_YT_CFG"    "$SNIP_PB/bitfit_st_fr15_ytvis.yaml" "$SEED" "model.initializer.init_mode=per_frame"
  done
fi

# ===== T3.2 — last-k blocks partial unfreeze =====
if is_wave "t32"; then
  echo "=== T3.2: Last-k partial unfreeze × MOVi-D × {SC, GCv1} × {k=2, k=4} × n=5 ==="
  for SEED in 0 1 2 3 4; do
    for K in 2 4; do
      submit "phaseB_movid_sc_lastk${K}_s${SEED}"      "$SC_MOVID_CFG" "$SNIP_PB/lastk_${K}_st_fr15.yaml" "$SEED"
      submit "phaseB_movid_gcv1_pf_lastk${K}_s${SEED}" "$GC_MOVID_CFG" "$SNIP_PB/lastk_${K}_st_fr15.yaml" "$SEED" "model.initializer.init_mode=per_frame"
    done
  done
fi

if ! is_wave "smoke" && ! is_wave "t31" && ! is_wave "t32"; then
  echo "Usage: bash $(basename $0) {smoke|t31|t32|all} [--live]"
  echo "  smoke: 500-step BitFit on gpu-debug"
  echo "  t31:   BitFit × 4 datasets × 2 archs × n=5  = 40 runs"
  echo "  t32:   Last-k {2,4} × MOVi-D × 2 archs × n=5 = 20 runs"
  echo "  all:   all 60 (plus smoke if --live)"
  echo "Save root: $OUTPUT_DIR"
  echo "experiment_group: $EXP_GROUP"
fi
