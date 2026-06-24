#!/bin/bash
# Optional "stronger scope" breadth waves at 100K, n=5 (+ MAE n=3).
# Extends three already-validated experiment families to more datasets:
#
#   Wave L (last-k breadth):  last-{2,4} partial unfreeze × {MOVi-C, MOVi-E,
#       YT-VIS} × {SC, GCv1} × 5 seeds = 60. Generalises phase_b T3.2
#       (which ran MOVi-D only). Lands in the phase_b save root.
#   Wave R (LoRA-LR breadth): off-canonical LoRA LR × {MOVi-C, MOVi-E,
#       YT-VIS} × {SC, GCv1} × 5 seeds = 60. Generalises phase_a T1.4
#       (which ran MOVi-D only). Lands in the phase_a save root.
#   Wave M (MAE learned):     YT-VIS × SC × MAE-ViT-B/16 × LEARNED predictor
#       × 3 seeds = 3. The skip_predictor=false sibling of the validated
#       phase_a t15_mae IDENTITY runs, to test identity≈learned for MAE.
#       Lands in the phase_a save root.
#
# DESIGN NOTES (verified before writing — see the pre-submit review):
#  - off-canonical LRs are per-dataset RELATIVE to each dataset's canonical
#    LoRA LR (0.1× and 10×), so the LR-robustness claim is apples-to-apples
#    with each dataset's canonical `st` cell:
#       MOVi-C / MOVi-E  canonical 4e-5 -> {4e-6, 4e-4}  (reuse phase_a movid_* snippets,
#                                                          non-encoder LR 4e-4 = MOVi base)
#       YT-VIS           canonical 8e-5 -> {8e-6, 8e-4}  (new ytvis_* snippets,
#                                                          non-encoder LR 8e-4 = YT-VIS base)
#  - GCv1 cells force model.initializer.init_mode=per_frame (the GCv1-canonical
#    setting used by the v2 grid). NOTE: the existing phase_a MOVi-D GC LoRA-LR
#    cells ran with the base default first_frame (no override); these breadth
#    cells deliberately use per_frame to match the v2 canonical `st` comparator.
#  - last-k YT-VIS uses LRs scaled to the YT-VIS 2× regime (encoder 8e-5,
#    non-encoder 8e-4); MOVi-C/E reuse the 4e-5/4e-4 phase_b snippets.
#  - MAE path is byte-identical to the validated phase_a t15_mae submit_mae()
#    except skip_predictor=false and the name token identity->learned.
#
# Usage:
#   bash scripts/launch_breadth_waves.sh                       # DRY-RUN (default)
#   bash scripts/launch_breadth_waves.sh lastk --live          # Wave L (60)
#   bash scripts/launch_breadth_waves.sh loralr --live         # Wave R (60)
#   bash scripts/launch_breadth_waves.sh mae --live            # Wave M (3)
#   bash scripts/launch_breadth_waves.sh all --live            # all 123
# Nothing is submitted without --live.

set -euo pipefail

LIVE=0
WAVES=""
for a in "$@"; do
  case "$a" in
    --live|-y|--yes) LIVE=1 ;;
    lastk|loralr|mae|all) WAVES="$WAVES $a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
WAVES="${WAVES:-help}"

TRITON_SLURM="triton_slurm.sh"
set +u
module load mamba 2>/dev/null || true
source activate slotcontrast 2>/dev/null || conda activate slotcontrast 2>/dev/null || true
set -u

PARTITION="gpu-h200-141g-ellis"
TIME_100K="1-12:00:00"
ACCOUNT="ellis_users"
COMMON_OVR="checkpoint_every_n_steps=25000"

# Save roots (reuse the existing phase_a / phase_b roots so the existing
# aggregator picks these up; regexes need the dataset list widened post-run).
PHASE_A_ROOT="/scratch/elec/t41020_egovla/slotcontrast_phase_a/v1_100k"
PHASE_B_ROOT="/scratch/elec/t41020_egovla/slotcontrast_phase_b/v1_100k"

# Base configs
SC_MOVIC_CFG="configs/slotcontrast/app/movi_c_slotcontrast_rescue15k.yaml"
SC_MOVIE_CFG="configs/slotcontrast/app/movi_e_slotcontrast_rescue15k.yaml"
SC_YT_CFG="configs/slotcontrast/app/ytvis2021_slotcontrast_rescue15k.yaml"
GC_MOVIC_CFG="configs/slotcontrast/app/movi_c_gcv1_rescue15k.yaml"
GC_MOVIE_CFG="configs/slotcontrast/app/movi_e_gcv1_rescue15k.yaml"
GC_YT_CFG="configs/slotcontrast/app/ytvis2021_gcv1_rescue15k.yaml"
SC_YT_BASELINE="configs/slotcontrast/app/ytvis2021_slotcontrast_baseline.yaml"

SNIP_PA="configs/slotcontrast/snippets/phase_a"
SNIP_PB="configs/slotcontrast/snippets/phase_b"

is_wave() { for w in $WAVES; do if [ "$w" = "$1" ] || [ "$w" = "all" ]; then return 0; fi; done; return 1; }
DRY=$([ "$LIVE" = "1" ] && echo "0" || echo "1")
[ "$DRY" = "1" ] && echo ">>> DRY-RUN MODE (no jobs submitted). Pass --live to submit. <<<"

# submit_snippet ROOT name cfg snippet seed [extra]
submit_snippet() {
  local ROOT=$1 NAME=$2 CFG=$3 SNIPPET=$4 SEED=$5 EXTRA="${6:-}"
  local LOGDIR="$ROOT/slurm_logs"; mkdir -p "$ROOT" "$LOGDIR"
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (root=$(basename "$ROOT") cfg=$(basename "$CFG") snippet=$(basename "$SNIPPET") seed=$SEED extra=$EXTRA)"
    return
  fi
  local EXP_GROUP; EXP_GROUP=$(basename "$(dirname "$ROOT")")  # slotcontrast_phase_{a,b}
  sbatch --job-name="$NAME" --account="$ACCOUNT" \
    --partition="$PARTITION" --time="$TIME_100K" --requeue \
    --export="ALL,OUTPUT_DIR=$ROOT" \
    --output="$LOGDIR/${NAME}_%j.out" --error="$LOGDIR/${NAME}_%j.err" \
    "$TRITON_SLURM" "$CFG" \
    "experiment_group=$EXP_GROUP" "experiment_name=$NAME" \
    "seed=$SEED" $COMMON_OVR $EXTRA \
    "--config_overrides_file=$SNIPPET"
}

# submit_mae ROOT name skip_predictor seed
submit_mae() {
  local ROOT=$1 NAME=$2 SP=$3 SEED=$4
  local LOGDIR="$ROOT/slurm_logs"; mkdir -p "$ROOT" "$LOGDIR"
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (mae-vit-base/16 input=224 num_patches=196 skip_pred=$SP seed=$SEED)"
    return
  fi
  local EXP_GROUP; EXP_GROUP=$(basename "$(dirname "$ROOT")")
  sbatch --job-name="$NAME" --account="$ACCOUNT" \
    --partition="$PARTITION" --time="$TIME_100K" --requeue \
    --export="ALL,OUTPUT_DIR=$ROOT" \
    --output="$LOGDIR/${NAME}_%j.out" --error="$LOGDIR/${NAME}_%j.err" \
    "$TRITON_SLURM" "$SC_YT_BASELINE" \
    "experiment_group=$(basename "$(dirname "$ROOT")")" "experiment_name=$NAME" \
    "seed=$SEED" \
    "globals.DINO_MODEL=vit_base_patch16_224.mae" \
    "globals.NUM_PATCHES=196" \
    "dataset.train_pipeline.transforms.input_size=224" \
    "dataset.val_pipeline.transforms.input_size=224" \
    "model.latent_processor.skip_predictor=$SP" \
    $COMMON_OVR
}

PF="model.initializer.init_mode=per_frame"

# ===== Wave L: last-k breadth (phase_b root) — 60 =====
if is_wave "lastk"; then
  echo "=== Wave L: last-k breadth × {MOVi-C, MOVi-E, YT-VIS} × {SC, GCv1} × {k=2,4} × n=5 — 60 ==="
  for SEED in 0 1 2 3 4; do
    for K in 2 4; do
      # MOVi-C / MOVi-E reuse the 4e-5/4e-4 phase_b snippet
      submit_snippet "$PHASE_B_ROOT" "phaseB_movic_sc_lastk${K}_s${SEED}"      "$SC_MOVIC_CFG" "$SNIP_PB/lastk_${K}_st_fr15.yaml" "$SEED"
      submit_snippet "$PHASE_B_ROOT" "phaseB_movie_sc_lastk${K}_s${SEED}"      "$SC_MOVIE_CFG" "$SNIP_PB/lastk_${K}_st_fr15.yaml" "$SEED"
      submit_snippet "$PHASE_B_ROOT" "phaseB_movic_gcv1_pf_lastk${K}_s${SEED}" "$GC_MOVIC_CFG" "$SNIP_PB/lastk_${K}_st_fr15.yaml" "$SEED" "$PF"
      submit_snippet "$PHASE_B_ROOT" "phaseB_movie_gcv1_pf_lastk${K}_s${SEED}" "$GC_MOVIE_CFG" "$SNIP_PB/lastk_${K}_st_fr15.yaml" "$SEED" "$PF"
      # YT-VIS uses the 8e-5/8e-4 YT-VIS-scaled snippet
      submit_snippet "$PHASE_B_ROOT" "phaseB_ytvis_sc_lastk${K}_s${SEED}"      "$SC_YT_CFG" "$SNIP_PB/lastk_${K}_st_fr15_ytvis.yaml" "$SEED"
      submit_snippet "$PHASE_B_ROOT" "phaseB_ytvis_gcv1_pf_lastk${K}_s${SEED}" "$GC_YT_CFG" "$SNIP_PB/lastk_${K}_st_fr15_ytvis.yaml" "$SEED" "$PF"
    done
  done
fi

# ===== Wave R: LoRA-LR breadth (phase_a root) — 60 =====
if is_wave "loralr"; then
  echo "=== Wave R: LoRA-LR breadth × {MOVi-C, MOVi-E, YT-VIS} × {SC, GCv1} × {lo,hi} × n=5 — 60 ==="
  for SEED in 0 1 2 3 4; do
    # MOVi-C / MOVi-E: canonical 4e-5 -> {4e-6, 4e-4}; reuse phase_a movid_* snippets (non-enc 4e-4 = MOVi base)
    for LR in 4e-6 4e-4; do
      submit_snippet "$PHASE_A_ROOT" "phaseA_movic_sc_lora_st_fr15_lr${LR}_s${SEED}"      "$SC_MOVIC_CFG" "$SNIP_PA/movid_sc_lora_st_fr15_lr${LR}.yaml" "$SEED"
      submit_snippet "$PHASE_A_ROOT" "phaseA_movie_sc_lora_st_fr15_lr${LR}_s${SEED}"      "$SC_MOVIE_CFG" "$SNIP_PA/movid_sc_lora_st_fr15_lr${LR}.yaml" "$SEED"
      submit_snippet "$PHASE_A_ROOT" "phaseA_movic_gcv1_pf_lora_st_fr15_lr${LR}_s${SEED}" "$GC_MOVIC_CFG" "$SNIP_PA/movid_gcv1_pf_lora_st_fr15_lr${LR}.yaml" "$SEED" "$PF"
      submit_snippet "$PHASE_A_ROOT" "phaseA_movie_gcv1_pf_lora_st_fr15_lr${LR}_s${SEED}" "$GC_MOVIE_CFG" "$SNIP_PA/movid_gcv1_pf_lora_st_fr15_lr${LR}.yaml" "$SEED" "$PF"
    done
    # YT-VIS: canonical 8e-5 -> {8e-6, 8e-4}; new arch-shared snippets (non-enc 8e-4 = YT-VIS base)
    for LR in 8e-6 8e-4; do
      submit_snippet "$PHASE_A_ROOT" "phaseA_ytvis_sc_lora_st_fr15_lr${LR}_s${SEED}"      "$SC_YT_CFG" "$SNIP_PA/ytvis_lora_st_fr15_lr${LR}.yaml" "$SEED"
      submit_snippet "$PHASE_A_ROOT" "phaseA_ytvis_gcv1_pf_lora_st_fr15_lr${LR}_s${SEED}" "$GC_YT_CFG" "$SNIP_PA/ytvis_lora_st_fr15_lr${LR}.yaml" "$SEED" "$PF"
    done
  done
fi

# ===== Wave M: MAE learned-predictor counterpart (phase_a root) — 3 =====
if is_wave "mae"; then
  echo "=== Wave M: MAE-ViT-B/16 × SC × YT-VIS × LEARNED predictor × n=3 — 3 ==="
  for SEED in 0 1 2; do
    submit_mae "$PHASE_A_ROOT" "phaseA_ytvis_sc_mae_native224_learned_s${SEED}" false "$SEED"
  done
fi

if [ "$WAVES" = "help" ]; then
  echo "Usage: bash $(basename "$0") {lastk|loralr|mae|all} [--live]"
  echo "  lastk : last-k breadth — 60 runs (phase_b root)"
  echo "  loralr: LoRA-LR breadth — 60 runs (phase_a root)"
  echo "  mae   : MAE learned counterpart — 3 runs (phase_a root)"
fi
