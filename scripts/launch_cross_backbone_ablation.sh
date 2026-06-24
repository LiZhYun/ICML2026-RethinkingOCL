#!/bin/bash
# Cross-backbone × predictor ablation for GC paper camera-ready appendix.
#
# Goal: repeat the original Table 3 predictor-removal test (SlotContrast,
# YouTube-VIS, n=3 seeds) across three pretrained backbones:
#   - DINOv1 ViT-B/16     (vit_base_patch16_224.dino)
#   - DINOv2 ViT-B/14     (vit_base_patch14_dinov2)       [default in main paper]
#   - DINOv3 ViT-B/16     (vit_base_patch16_dinov3)
#
# Each backbone × {learned predictor, identity Qt=St-1} × n=3 seeds = 18 runs.
#
# Save root: /scratch/elec/t41020_egovla/slotcontrast_cross_backbone/v1_100k/
# experiment_group: slotcontrast_cross_backbone
#
# Usage:
#   bash scripts/launch_cross_backbone_ablation.sh smoke --live    # 1k-step gpu-debug
#   bash scripts/launch_cross_backbone_ablation.sh all   --live    # 18-run 100K grid
#   bash scripts/launch_cross_backbone_ablation.sh all              # DRY-RUN (default)

set -euo pipefail

# Parse args (same convention as launch_full_100k_v2.sh).
LIVE=0
WAVES=""
for a in "$@"; do
  case "$a" in
    --live|-y|--yes) LIVE=1 ;;
    smoke|all) WAVES="$WAVES $a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
WAVES="${WAVES:-help}"

TRITON_SLURM="triton_slurm.sh"

set +u
module load mamba 2>/dev/null || true
source activate slotcontrast 2>/dev/null || conda activate slotcontrast 2>/dev/null || true
set -u

export OUTPUT_DIR="/scratch/elec/t41020_egovla/slotcontrast_cross_backbone/v1_100k"
SLURM_LOG_DIR="$OUTPUT_DIR/slurm_logs"
mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"

EXP_GROUP="slotcontrast_cross_backbone"
BASE_CFG="configs/slotcontrast/app/ytvis2021_slotcontrast_baseline.yaml"

PARTITION="gpu-h200-141g-ellis"
TIME_100K="1-12:00:00"
ACCOUNT="ellis_users"
COMMON_OVR="checkpoint_every_n_steps=25000"

is_wave() { for w in $WAVES; do if [ "$w" = "$1" ] || [ "$w" = "all" ]; then return 0; fi; done; return 1; }
DRY=$([ "$LIVE" = "1" ] && echo "0" || echo "1")
if [ "$DRY" = "1" ]; then
  echo ">>> DRY-RUN MODE (no jobs will be submitted). Pass --live to submit. <<<"
fi

# submit name backbone num_patches input_size skip_predictor seed
submit() {
  local NAME=$1 BB=$2 NP=$3 IS=$4 SP=$5 SEED=$6
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (backbone=$BB num_patches=$NP input=$IS skip_pred=$SP seed=$SEED)"
    return
  fi
  sbatch --job-name="$NAME" --account="$ACCOUNT" \
    --partition="$PARTITION" --time="$TIME_100K" \
    --requeue \
    --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
    --output="$SLURM_LOG_DIR/${NAME}_%j.out" \
    --error="$SLURM_LOG_DIR/${NAME}_%j.err" \
    "$TRITON_SLURM" "$BASE_CFG" \
    "experiment_group=$EXP_GROUP" "experiment_name=$NAME" \
    "seed=$SEED" \
    "globals.DINO_MODEL=$BB" \
    "globals.NUM_PATCHES=$NP" \
    "dataset.train_pipeline.transforms.input_size=$IS" \
    "dataset.val_pipeline.transforms.input_size=$IS" \
    "model.latent_processor.skip_predictor=$SP" \
    $COMMON_OVR
}

# ----- SMOKE: 1 quick gpu-debug job to verify pipeline works on a new backbone -----
if is_wave "smoke"; then
  echo "=== SMOKE: 1k-step pilot DINOv3 + identity predictor on gpu-debug ==="
  if [ "$DRY" = "1" ]; then
    echo "DRY: smoke_dinov3_identity_v1_1k"
  else
    sbatch --job-name="smoke_dinov3_identity_v1_1k" --account="aalto_users" \
      --partition="gpu-debug" --time="0-00:30:00" \
      --requeue \
      --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
      --output="$SLURM_LOG_DIR/smoke_dinov3_identity_v1_1k_%j.out" \
      --error="$SLURM_LOG_DIR/smoke_dinov3_identity_v1_1k_%j.err" \
      "$TRITON_SLURM" "$BASE_CFG" \
      "experiment_group=${EXP_GROUP}_smoke" "experiment_name=smoke_dinov3_identity_v1_1k" \
      "seed=0" "trainer.max_steps=1000" "trainer.val_check_interval=500" \
      "checkpoint_every_n_steps=500" \
      "globals.DINO_MODEL=vit_base_patch16_dinov3" \
      "globals.NUM_PATCHES=1024" \
      "dataset.train_pipeline.transforms.input_size=512" \
      "dataset.val_pipeline.transforms.input_size=512" \
      "model.latent_processor.skip_predictor=true"
  fi
fi

# ----- ALL: 18-run grid (3 backbones × 2 predictor variants × n=3 seeds) -----
if is_wave "all"; then
  echo "=== Cross-backbone × predictor grid: 3 backbones × 2 variants × n=3 = 18 runs ==="
  for SEED in 0 1 2; do
    # DINOv1: patch=16, input=512, NUM_PATCHES=1024
    submit "ytvis_sc_dinov1_learned_s${SEED}"  "vit_base_patch16_224.dino" 1024 512 false "$SEED"
    submit "ytvis_sc_dinov1_identity_s${SEED}" "vit_base_patch16_224.dino" 1024 512 true  "$SEED"
    # DINOv2: patch=14, input=518, NUM_PATCHES=1369 (main paper default)
    submit "ytvis_sc_dinov2_learned_s${SEED}"  "vit_base_patch14_dinov2"   1369 518 false "$SEED"
    submit "ytvis_sc_dinov2_identity_s${SEED}" "vit_base_patch14_dinov2"   1369 518 true  "$SEED"
    # DINOv3: patch=16, input=512, NUM_PATCHES=1024
    submit "ytvis_sc_dinov3_learned_s${SEED}"  "vit_base_patch16_dinov3"   1024 512 false "$SEED"
    submit "ytvis_sc_dinov3_identity_s${SEED}" "vit_base_patch16_dinov3"   1024 512 true  "$SEED"
  done
fi

if ! is_wave "smoke" && ! is_wave "all"; then
  echo "Usage: bash $(basename $0) {smoke|all} [--live]"
  echo "  smoke: 1 quick gpu-debug job (~5 min) — DINOv3 + identity + 1K steps"
  echo "  all:   18-run grid (3 backbones × {learned, identity} × n=3 seeds at 100K)"
  echo ""
  echo "Save root: $OUTPUT_DIR"
  echo "Slurm logs: $SLURM_LOG_DIR"
  echo "experiment_group: $EXP_GROUP"
  echo "WITHOUT --live the script is in DRY-RUN MODE."
fi
