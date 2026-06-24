#!/bin/bash
# Phase-A grid (Tier 1 from the post-nightmare-review action plan).
#
# Cells:
#   T1.1  YT-VIS fullft LR fix:
#           ytvis × {sc, gcv1_pf} × fullft_fix × n=5     = 10 runs
#   T1.2  MOVi-D fullft LR sweep at {1e-5, 1e-4, 4e-4}:
#           movid × {sc, gcv1_pf} × 3 LRs × n=5         = 30 runs
#   T1.3  DINOv3 cross-backbone re-run at native 224 input:
#           ytvis × sc × dinov3 × {identity, learned} × n=5 = 10 runs
#
# Total: 50 100K-step runs.
# Save root: /scratch/elec/t41020_egovla/slotcontrast_phase_a/v1_100k/
# experiment_group: slotcontrast_phase_a
#
# Usage:
#   bash scripts/launch_phase_a.sh smoke --live   # 1 quick gpu-debug job
#   bash scripts/launch_phase_a.sh all   --live   # all 50 runs
#   bash scripts/launch_phase_a.sh all             # DRY-RUN (default)

set -euo pipefail

LIVE=0
WAVES=""
for a in "$@"; do
  case "$a" in
    --live|-y|--yes) LIVE=1 ;;
    smoke|t11|t12|t13|t12_extra|t11_lrsweep|t14_lora_lr|t14_lora_lr_scaffold|t14_lora_lr_n5|t15_mae|all) WAVES="$WAVES $a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
WAVES="${WAVES:-help}"

TRITON_SLURM="triton_slurm.sh"
SNIP_PA="configs/slotcontrast/snippets/phase_a"

set +u
module load mamba 2>/dev/null || true
source activate slotcontrast 2>/dev/null || conda activate slotcontrast 2>/dev/null || true
set -u

export OUTPUT_DIR="/scratch/elec/t41020_egovla/slotcontrast_phase_a/v1_100k"
SLURM_LOG_DIR="$OUTPUT_DIR/slurm_logs"
mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"

EXP_GROUP="slotcontrast_phase_a"
PARTITION="gpu-h200-141g-ellis"
TIME_100K="1-12:00:00"
ACCOUNT="ellis_users"
COMMON_OVR="checkpoint_every_n_steps=25000"

# Base configs per dataset × arch (same as the v2 launcher uses)
SC_MOVIC_CFG="configs/slotcontrast/app/movi_c_slotcontrast_rescue15k.yaml"
SC_MOVID_CFG="configs/slotcontrast/app/movi_d_slotcontrast_rescue15k.yaml"
SC_MOVIE_CFG="configs/slotcontrast/app/movi_e_slotcontrast_rescue15k.yaml"
SC_YT_CFG="configs/slotcontrast/app/ytvis2021_slotcontrast_rescue15k.yaml"
GC_MOVIC_CFG="configs/slotcontrast/app/movi_c_gcv1_rescue15k.yaml"
GC_MOVID_CFG="configs/slotcontrast/app/movi_d_gcv1_rescue15k.yaml"
GC_MOVIE_CFG="configs/slotcontrast/app/movi_e_gcv1_rescue15k.yaml"
GC_YT_CFG="configs/slotcontrast/app/ytvis2021_gcv1_rescue15k.yaml"
SC_YT_BASELINE="configs/slotcontrast/app/ytvis2021_slotcontrast_baseline.yaml"

is_wave() { for w in $WAVES; do if [ "$w" = "$1" ] || [ "$w" = "all" ]; then return 0; fi; done; return 1; }
DRY=$([ "$LIVE" = "1" ] && echo "0" || echo "1")
if [ "$DRY" = "1" ]; then
  echo ">>> DRY-RUN MODE (no jobs will be submitted). Pass --live to submit. <<<"
fi

# submit_snippet name cfg snippet_path seed
submit_snippet() {
  local NAME=$1 CFG=$2 SNIPPET=$3 SEED=$4
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (cfg=$(basename "$CFG") snippet=$(basename "$SNIPPET") seed=$SEED)"
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
    "seed=$SEED" $COMMON_OVR \
    "--config_overrides_file=$SNIPPET"
}

# submit_dinov3 name cfg input_size num_patches skip_predictor seed
submit_dinov3() {
  local NAME=$1 INPUT=$2 NP=$3 SP=$4 SEED=$5
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (dinov3 input=$INPUT num_patches=$NP skip_pred=$SP seed=$SEED)"
    return
  fi
  sbatch --job-name="$NAME" --account="$ACCOUNT" \
    --partition="$PARTITION" --time="$TIME_100K" \
    --requeue \
    --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
    --output="$SLURM_LOG_DIR/${NAME}_%j.out" \
    --error="$SLURM_LOG_DIR/${NAME}_%j.err" \
    "$TRITON_SLURM" "$SC_YT_BASELINE" \
    "experiment_group=$EXP_GROUP" "experiment_name=$NAME" \
    "seed=$SEED" \
    "globals.DINO_MODEL=vit_base_patch16_dinov3" \
    "globals.NUM_PATCHES=$NP" \
    "dataset.train_pipeline.transforms.input_size=$INPUT" \
    "dataset.val_pipeline.transforms.input_size=$INPUT" \
    "model.latent_processor.skip_predictor=$SP" \
    $COMMON_OVR
}

# ===== SMOKE =====
if is_wave "smoke"; then
  echo "=== SMOKE: 1k-step pilot — MOVi-D × SC × fullft × lr=1e-4 on gpu-debug ==="
  if [ "$DRY" = "1" ]; then
    echo "DRY: smoke_movid_sc_fullft_lr1e-4_1k"
  else
    sbatch --job-name="smoke_phaseA_1k" --account="aalto_users" \
      --partition="gpu-debug" --time="0-00:30:00" \
      --requeue \
      --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
      --output="$SLURM_LOG_DIR/smoke_phaseA_1k_%j.out" \
      --error="$SLURM_LOG_DIR/smoke_phaseA_1k_%j.err" \
      "$TRITON_SLURM" "$SC_MOVID_CFG" \
      "experiment_group=${EXP_GROUP}_smoke" \
      "experiment_name=smoke_movid_sc_fullft_lr1e-4_1k" \
      "seed=0" "trainer.max_steps=1000" "trainer.val_check_interval=500" \
      "checkpoint_every_n_steps=500" \
      "--config_overrides_file=$SNIP_PA/movid_sc_fullft_lr1e-4.yaml"
  fi
fi

# ===== T1.1 — YT-VIS fullft LR fix =====
if is_wave "t11"; then
  echo "=== T1.1: YT-VIS fullft LR fix × n=5 seeds ==="
  for SEED in 0 1 2 3 4; do
    submit_snippet "phaseA_ytvis_sc_fullft_fix_s${SEED}" \
      "$SC_YT_CFG" "$SNIP_PA/ytvis_sc_fullft_fix.yaml" "$SEED"
    submit_snippet "phaseA_ytvis_gcv1_pf_fullft_fix_s${SEED}" \
      "$GC_YT_CFG" "$SNIP_PA/ytvis_gcv1_pf_fullft_fix.yaml" "$SEED"
  done
fi

# ===== T1.2 — MOVi-D fullft LR sweep =====
if is_wave "t12"; then
  echo "=== T1.2: MOVi-D fullft LR sweep × n=5 seeds ==="
  for SEED in 0 1 2 3 4; do
    for LR in 1e-5 1e-4 4e-4; do
      submit_snippet "phaseA_movid_sc_fullft_lr${LR}_s${SEED}" \
        "$SC_MOVID_CFG" "$SNIP_PA/movid_sc_fullft_lr${LR}.yaml" "$SEED"
      submit_snippet "phaseA_movid_gcv1_pf_fullft_lr${LR}_s${SEED}" \
        "$GC_MOVID_CFG" "$SNIP_PA/movid_gcv1_pf_fullft_lr${LR}.yaml" "$SEED"
    done
  done
fi

# ===== T1.2 extra — MOVi-C and MOVi-E fullft @ lr=1e-5 =====
# Round-2 reviewer follow-up: extend the LR-conditional fullft picture
# from MOVi-D to MOVi-C and MOVi-E. If LoRA-rescue beats fullft@1e-5
# decisively on these datasets, the v2 grid's "fullft fails" headline
# survives once we tune the LR. If fullft@1e-5 is competitive (as on
# SC × MOVi-D), the rescue claim weakens on those cells too.
#
# Reuses the existing movid_*_fullft_lr1e-5.yaml snippets (dataset-
# agnostic: just sets encoder lr=1e-5, non-encoder lr=4e-4 = MOVi base).
if is_wave "t12_extra"; then
  echo "=== T1.2-extra: MOVi-{C,E} fullft @ lr=1e-5 × n=5 seeds ==="
  for SEED in 0 1 2 3 4; do
    submit_snippet "phaseA_movic_sc_fullft_lr1e-5_s${SEED}" \
      "$SC_MOVIC_CFG" "$SNIP_PA/movid_sc_fullft_lr1e-5.yaml" "$SEED"
    submit_snippet "phaseA_movic_gcv1_pf_fullft_lr1e-5_s${SEED}" \
      "$GC_MOVIC_CFG" "$SNIP_PA/movid_gcv1_pf_fullft_lr1e-5.yaml" "$SEED"
    submit_snippet "phaseA_movie_sc_fullft_lr1e-5_s${SEED}" \
      "$SC_MOVIE_CFG" "$SNIP_PA/movid_sc_fullft_lr1e-5.yaml" "$SEED"
    submit_snippet "phaseA_movie_gcv1_pf_fullft_lr1e-5_s${SEED}" \
      "$GC_MOVIE_CFG" "$SNIP_PA/movid_gcv1_pf_fullft_lr1e-5.yaml" "$SEED"
  done
fi

# ===== T1.4-redo — LoRA encoder LR sweep with ST+featrec=1.5 scaffold (Round-34 fix) =====
# Round-33 codex caught that the original t14_lora_lr snippets inherited the
# base config's native predictor (TransformerEncoder on SC, HungarianPredictor
# on GCv1) and featrec=1.0, not the rescue scaffold's ST-SoftIdent + featrec=1.5
# that the canonical LoRA-rescue (`st`) and tuned fullft@1e-5 comparators use.
# This redo explicitly overrides the predictor + loss_weights to match the
# v2 `_st_` scaffold so the LR sweep is apples-to-apples.
if is_wave "t14_lora_lr_scaffold"; then
  echo "=== T1.4-redo: LoRA encoder-LR sweep with ST+fr15 scaffold × n=3 seeds (Round-34) ==="
  for SEED in 0 1 2; do
    for LR in 4e-6 4e-4; do
      submit_snippet "phaseA_movid_sc_lora_st_fr15_lr${LR}_s${SEED}" \
        "$SC_MOVID_CFG" "$SNIP_PA/movid_sc_lora_st_fr15_lr${LR}.yaml" "$SEED"
      submit_snippet "phaseA_movid_gcv1_pf_lora_st_fr15_lr${LR}_s${SEED}" \
        "$GC_MOVID_CFG" "$SNIP_PA/movid_gcv1_pf_lora_st_fr15_lr${LR}.yaml" "$SEED"
    done
  done
fi

# ===== T1.4-n5 — Upgrade scaffolded LoRA LR sweep to n=5 (Round-35 follow-up) =====
# Codex Round 35 prescribed n=5 instead of n=3 for journal strength. The
# scaffolded snippets already exist; just submit seeds 3 and 4 for the 4
# off-canonical cells = 8 new cells.
if is_wave "t14_lora_lr_n5"; then
  echo "=== T1.4-n5: scaffolded LoRA LR sweep seeds 3,4 (Round-35 follow-up) ==="
  for SEED in 3 4; do
    for LR in 4e-6 4e-4; do
      submit_snippet "phaseA_movid_sc_lora_st_fr15_lr${LR}_s${SEED}" \
        "$SC_MOVID_CFG" "$SNIP_PA/movid_sc_lora_st_fr15_lr${LR}.yaml" "$SEED"
      submit_snippet "phaseA_movid_gcv1_pf_lora_st_fr15_lr${LR}_s${SEED}" \
        "$GC_MOVID_CFG" "$SNIP_PA/movid_gcv1_pf_lora_st_fr15_lr${LR}.yaml" "$SEED"
    done
  done
fi

# ===== T1.5 — MAE-ViT-B/16 third backbone (Round-33 FIX-14) =====
# Cross-backbone appendix currently DINOv1/v2 only (DINOv3 dropped in §6.4).
# Adds MAE-pretrained ViT-B/16 — a different SSL paradigm (masked image
# modeling vs DINO's contrastive self-distillation) — on SC × YT-VIS ×
# identity predictor × n=3 to test whether SC's "identity ≈ learned"
# pattern survives a different pretrained visual representation. timm
# model: vit_base_patch16_224.mae, native 224 input → 196 patches.
submit_mae() {
  local NAME=$1 SP=$2 SEED=$3
  if [ "$DRY" = "1" ]; then
    echo "DRY: $NAME (mae-vit-base/16 input=224 num_patches=196 skip_pred=$SP seed=$SEED)"
    return
  fi
  sbatch --job-name="$NAME" --account="$ACCOUNT" \
    --partition="$PARTITION" --time="$TIME_100K" \
    --requeue \
    --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
    --output="$SLURM_LOG_DIR/${NAME}_%j.out" \
    --error="$SLURM_LOG_DIR/${NAME}_%j.err" \
    "$TRITON_SLURM" "$SC_YT_BASELINE" \
    "experiment_group=$EXP_GROUP" "experiment_name=$NAME" \
    "seed=$SEED" \
    "globals.DINO_MODEL=vit_base_patch16_224.mae" \
    "globals.NUM_PATCHES=196" \
    "dataset.train_pipeline.transforms.input_size=224" \
    "dataset.val_pipeline.transforms.input_size=224" \
    "model.latent_processor.skip_predictor=$SP" \
    $COMMON_OVR
}
if is_wave "t15_mae"; then
  echo "=== T1.5: MAE-ViT-B/16 cross-backbone × SC × YT-VIS × identity × n=3 (Round-33 FIX-14) ==="
  for SEED in 0 1 2; do
    submit_mae "phaseA_ytvis_sc_mae_native224_identity_s${SEED}" true "$SEED"
  done
fi

# ===== T1.1-extra — YT-VIS fullft LR sweep at lr=1e-5 (Round-32 FIX-4) =====
# T1.1 established fullft fails on YT-VIS at the LoRA-matched encoder lr (8e-5).
# Reviewer flagged this is matched-LR, not LR-tuned, so the v5 abstract's
# "tuned fullft fails on YT-VIS" claim is unsupported. This wave reruns YT-VIS
# fullft at the MOVi sweet-spot lr (1e-5) × n=5 to test whether YT-VIS shows
# the same LR-conditional recovery or stays LR-robustly broken.
if is_wave "t11_lrsweep"; then
  echo "=== T1.1-extra: YT-VIS fullft @ lr=1e-5 × n=5 seeds ==="
  for SEED in 0 1 2 3 4; do
    submit_snippet "phaseA_ytvis_sc_fullft_lr1e-5_s${SEED}" \
      "$SC_YT_CFG" "$SNIP_PA/ytvis_sc_fullft_lr1e-5.yaml" "$SEED"
    submit_snippet "phaseA_ytvis_gcv1_pf_fullft_lr1e-5_s${SEED}" \
      "$GC_YT_CFG" "$SNIP_PA/ytvis_gcv1_pf_fullft_lr1e-5.yaml" "$SEED"
  done
fi

# ===== T1.4 — LoRA encoder LR sweep (Round-33 FIX-12) =====
# T1.2 showed fullft has a narrow encoder-LR sweet spot at 1e-5. To support
# the "LoRA is LR-robust" claim rigorously, we sweep LoRA's encoder LR by
# ±10× from its canonical 4e-5 setting: {4e-6, 4e-4} on MOVi-D × {SC, GCv1}
# × n=3 = 12 cells. The 4e-4 cell is the LR where fullft collapses — if
# LoRA still works there, the LR-robustness claim is empirically supported.
if is_wave "t14_lora_lr"; then
  echo "=== T1.4: LoRA encoder LR sweep × n=3 seeds (Round-33 FIX-12) ==="
  for SEED in 0 1 2; do
    for LR in 4e-6 4e-4; do
      submit_snippet "phaseA_movid_sc_lora_lr${LR}_s${SEED}" \
        "$SC_MOVID_CFG" "$SNIP_PA/movid_sc_lora_lr${LR}.yaml" "$SEED"
      submit_snippet "phaseA_movid_gcv1_pf_lora_lr${LR}_s${SEED}" \
        "$GC_MOVID_CFG" "$SNIP_PA/movid_gcv1_pf_lora_lr${LR}.yaml" "$SEED"
    done
  done
fi

# ===== T1.3 — DINOv3 native 224 cross-backbone =====
if is_wave "t13"; then
  echo "=== T1.3: DINOv3 native 224 cross-backbone × n=5 seeds ==="
  for SEED in 0 1 2 3 4; do
    submit_dinov3 "phaseA_ytvis_sc_dinov3_native224_learned_s${SEED}" \
      224 196 false "$SEED"
    submit_dinov3 "phaseA_ytvis_sc_dinov3_native224_identity_s${SEED}" \
      224 196 true "$SEED"
  done
fi

if ! is_wave "smoke" && ! is_wave "t11" && ! is_wave "t12" && ! is_wave "t13" && ! is_wave "t12_extra" && ! is_wave "t11_lrsweep" && ! is_wave "t14_lora_lr"; then
  echo "Usage: bash $(basename $0) {smoke|t11|t11_lrsweep|t12|t13|t12_extra|t14_lora_lr|all} [--live]"
  echo "  t14_lora_lr:  MOVi-D LoRA encoder-LR sweep × {4e-6, 4e-4} × 2 archs × n=3 = 12 runs"
  echo "  smoke:        1k-step MOVi-D × SC × fullft × lr=1e-4 on gpu-debug"
  echo "  t11:          YT-VIS fullft LR fix × n=5 × 2 archs            = 10 runs"
  echo "  t11_lrsweep:  YT-VIS fullft @ lr=1e-5 × n=5 × 2 archs         = 10 runs"
  echo "  t12:          MOVi-D fullft LR sweep × 3 LRs × 2 archs × n=5  = 30 runs"
  echo "  t13:          DINOv3 native 224 × {identity, learned} × n=5   = 10 runs"
  echo "  t12_extra:    MOVi-{C,E} fullft @ lr=1e-5 × 2 archs × n=5     = 20 runs"
  echo "  all:          all of the above"
  echo ""
  echo "Save root: $OUTPUT_DIR"
  echo "experiment_group: $EXP_GROUP"
fi
