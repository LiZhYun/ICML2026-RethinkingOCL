#!/bin/bash
# v2 100K experiment grid — exhaustive, with checkpoints saved.
#
# Total: 375 training runs at 100K steps × n=5 seeds {0,1,2,3,4}.
# Save root: /scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k/
# Checkpoints every 25K steps (4 ckpts per run).
# H200 Ellis-priority partition only (no B300; sm_100 incompatible with
# the bundled PyTorch 2.6 / CUDA 12.6 build).
#
# Variants per architecture:
#   GCv1 (init_mode: per_frame, slot-slot contrastive loss enabled):
#     frozen, ST, Hungarian, no-ST (soft Sinkhorn), no-matching, full-FT, featrec=0
#   SC (init_mode: first_frame, slot-slot contrastive loss enabled):
#     frozen, ST, Hungarian [NEW], default-pred (TransformerEncoder), no-ST,
#     no-matching, full-FT, featrec=0
# (Both archs use loss_ss: 0.5 + temperature 0.1 — verified from saved
# settings.yaml across every cell.)
#
# Ablations (GC × MOVi-C only):
#   featrec sweep {0.5, 1.0, 2.0}
#   LoRA rank {4, 16, 32}
#   R3 unfreeze step {0, 2500, 10000}
#
# D1-light (SC × {MOVi-D, MOVi-E} with slot-slot loss disabled):
#   {ST, default-pred, no-matching} × n=5
#
# Usage:
#   bash scripts/launch_full_100k_v2.sh wave1                # DRY-RUN (default; shows what would submit)
#   bash scripts/launch_full_100k_v2.sh wave1 --live         # actually submit Wave 1
#   bash scripts/launch_full_100k_v2.sh smoke --live         # 1k-step gpu-debug smoke
#   bash scripts/launch_full_100k_v2.sh all --live           # submit all 4 waves
# Default behaviour is DRY: nothing is submitted unless `--live` is one of the args.

set -euo pipefail

# Parse args: extract --live, leave wave selectors in $WAVES.
LIVE=0
WAVES=""
for a in "$@"; do
  case "$a" in
    --live|-y|--yes) LIVE=1 ;;
    smoke|wave1|wave2|wave3|wave4|all) WAVES="$WAVES $a" ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
WAVES="${WAVES:-help}"

TRITON_SLURM="triton_slurm.sh"
SNIP=configs/slotcontrast/snippets
SNIP_OUT_DIR="configs/slotcontrast/snippets/v2_generated"
mkdir -p "$SNIP_OUT_DIR"

set +u
module load mamba 2>/dev/null || true
source activate slotcontrast 2>/dev/null || conda activate slotcontrast 2>/dev/null || true
set -u

# Save folder propagates to triton_slurm.sh. We pass it via sbatch --export so
# routing does not depend on site defaults.
export OUTPUT_DIR="/scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k"
SLURM_LOG_DIR="$OUTPUT_DIR/slurm_logs"
mkdir -p "$OUTPUT_DIR" "$SLURM_LOG_DIR"

EXP_GROUP="slotcontrast_v2_100k"

# Dataset configs
GC_MC_CFG="configs/slotcontrast/app/movi_c_gcv1_rescue15k.yaml"
GC_MD_CFG="configs/slotcontrast/app/movi_d_gcv1_rescue15k.yaml"
GC_ME_CFG="configs/slotcontrast/app/movi_e_gcv1_rescue15k.yaml"
GC_YT_CFG="configs/slotcontrast/app/ytvis2021_gcv1_rescue15k.yaml"
SC_MC_CFG="configs/slotcontrast/app/movi_c_slotcontrast_rescue15k.yaml"
SC_MD_CFG="configs/slotcontrast/app/movi_d_slotcontrast_rescue15k.yaml"
SC_ME_CFG="configs/slotcontrast/app/movi_e_slotcontrast_rescue15k.yaml"
SC_YT_CFG="configs/slotcontrast/app/ytvis2021_slotcontrast_rescue15k.yaml"

# Variant snippets (existing)
SN_ST="$SNIP/softident_st_fr15.yaml"
SN_HUNG_GC="$SNIP/hungarian_lora_fr15.yaml"
SN_NOST_GC="$SNIP/soft_identity_predictor_featrec15.yaml"
SN_NOST_SC="$SNIP/sc_softident_no_st_fr15.yaml"
SN_NOMATCH="$SNIP/nomatch_lora_fr15.yaml"
SN_FROZEN="$SNIP/frozen_15k.yaml"
SN_DEFAULT_PRED="$SNIP/sc_default_predictor_fr15.yaml"

# Variant snippets (new in v2)
SN_FULLFT="$SNIP/fullft_st_fr15.yaml"
SN_FEATREC0="$SNIP/featrec_w0_st.yaml"
SN_HUNG_SC="$SNIP/sc_hung_lora_fr15.yaml"

# Ablation snippets (already 100K-baked)
SN_ABL_FW0p5="$SNIP/100k_generated/ablation_featrec_w0p5_movic_gcv1_100k.yaml"
SN_ABL_FW1p0="$SNIP/100k_generated/ablation_featrec_w1p0_movic_gcv1_100k.yaml"
SN_ABL_FW2p0="$SNIP/100k_generated/ablation_featrec_w2p0_movic_gcv1_100k.yaml"
SN_ABL_LORA_R4="$SNIP/100k_generated/ablation_lora_r4_a8_movic_gcv1_100k.yaml"
SN_ABL_LORA_R16="$SNIP/100k_generated/ablation_lora_r16_a32_movic_gcv1_100k.yaml"
SN_ABL_LORA_R32="$SNIP/100k_generated/ablation_lora_r32_a64_movic_gcv1_100k.yaml"
SN_ABL_R3_0="$SNIP/100k_generated/ablation_r3step_0_movic_gcv1_100k.yaml"
SN_ABL_R3_2500="$SNIP/100k_generated/ablation_r3step_2500_movic_gcv1_100k.yaml"
SN_ABL_R3_10K="$SNIP/100k_generated/ablation_r3step_10000_movic_gcv1_100k.yaml"

# D1-light snippets (SC w/o slot-slot loss; same content across cells).
SN_D1_ST="$SNIP/100k_generated/ablation_sc_st_no_ss_movid_100k.yaml"
SN_D1_DEFAULT="$SNIP/100k_generated/ablation_sc_default_no_ss_movid_100k.yaml"
SN_D1_NOMATCH="$SNIP/100k_generated/ablation_sc_nomatch_no_ss_movid_100k.yaml"

# Submission infra
PARTITION="gpu-h200-141g-ellis"
TIME_100K="1-12:00:00"     # 36h walltime; ~25h expected at 100K
ACCOUNT="ellis_users"
COMMON_OVR="checkpoint_every_n_steps=25000"   # 4 ckpts per run

is_wave() { for w in $WAVES; do if [ "$w" = "$1" ] || [ "$w" = "all" ]; then return 0; fi; done; return 1; }
DRY=$([ "$LIVE" = "1" ] && echo "0" || echo "1")
if [ "$DRY" = "1" ]; then
  echo ">>> DRY-RUN MODE (no jobs will be submitted). Pass --live to submit. <<<"
fi

SEEDS="0 1 2 3 4"

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

# ----- SMOKE -----
if is_wave "smoke"; then
  echo "=== SMOKE: 1k-step pilot SC MOVi-C ST-SoftIdent+fr15 on gpu-debug ==="
  SNIPPET=$(build_snippet "$SC_MC_CFG" "$SN_ST" "smoke_movic_sc_st_v2_1k")
  if [ "$DRY" = "1" ]; then
    echo "DRY: smoke job built snippet at $SNIPPET"
  else
    sbatch --job-name="smoke_movic_sc_st_v2_1k" --account="aalto_users" \
      --partition="gpu-debug" --time="0-00:30:00" \
      --requeue \
      --export="ALL,OUTPUT_DIR=$OUTPUT_DIR" \
      --output="$SLURM_LOG_DIR/smoke_movic_sc_st_v2_1k_%j.out" \
      --error="$SLURM_LOG_DIR/smoke_movic_sc_st_v2_1k_%j.err" \
      "$TRITON_SLURM" "$SC_MC_CFG" \
      "experiment_group=${EXP_GROUP}_smoke" "experiment_name=smoke_movic_sc_st_v2_1k" \
      "seed=0" "trainer.max_steps=1000" "trainer.val_check_interval=500" \
      "checkpoint_every_n_steps=500" \
      "--config_overrides_file=$SNIPPET"
  fi
fi

# ----- WAVE 1: anchors (frozen + ST × 8 cells × n=5) — 80 runs -----
if is_wave "wave1"; then
  echo "=== WAVE 1: Frozen + ST anchors @ 100K, n=5 — 80 runs ==="
  for SEED in $SEEDS; do
    # GCv1 frozen
    submit "movic_gcv1_pf_frozen_v2_s${SEED}"  "$GC_MC_CFG" "$SN_FROZEN" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_frozen_v2_s${SEED}"  "$GC_MD_CFG" "$SN_FROZEN" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_frozen_v2_s${SEED}"  "$GC_ME_CFG" "$SN_FROZEN" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_frozen_v2_s${SEED}"  "$GC_YT_CFG" "$SN_FROZEN" "$SEED" "model.initializer.init_mode=per_frame"
    # SC frozen
    submit "movic_sc_frozen_v2_s${SEED}" "$SC_MC_CFG" "$SN_FROZEN" "$SEED"
    submit "movid_sc_frozen_v2_s${SEED}" "$SC_MD_CFG" "$SN_FROZEN" "$SEED"
    submit "movie_sc_frozen_v2_s${SEED}" "$SC_ME_CFG" "$SN_FROZEN" "$SEED"
    submit "ytvis_sc_frozen_v2_s${SEED}" "$SC_YT_CFG" "$SN_FROZEN" "$SEED"
    # GCv1 ST
    submit "movic_gcv1_pf_st_v2_s${SEED}"  "$GC_MC_CFG" "$SN_ST" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_st_v2_s${SEED}"  "$GC_MD_CFG" "$SN_ST" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_st_v2_s${SEED}"  "$GC_ME_CFG" "$SN_ST" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_st_v2_s${SEED}"  "$GC_YT_CFG" "$SN_ST" "$SEED" "model.initializer.init_mode=per_frame"
    # SC ST
    submit "movic_sc_st_v2_s${SEED}" "$SC_MC_CFG" "$SN_ST" "$SEED"
    submit "movid_sc_st_v2_s${SEED}" "$SC_MD_CFG" "$SN_ST" "$SEED"
    submit "movie_sc_st_v2_s${SEED}" "$SC_ME_CFG" "$SN_ST" "$SEED"
    submit "ytvis_sc_st_v2_s${SEED}" "$SC_YT_CFG" "$SN_ST" "$SEED"
  done
fi

# ----- WAVE 2: predictor counterfactuals — 140 runs -----
# GC × {Hung, noST, nomatch} = 60
# SC × {Hung [NEW], default_pred, noST, nomatch} = 80
if is_wave "wave2"; then
  echo "=== WAVE 2: Predictor counterfactuals @ 100K, n=5 — 140 runs ==="
  for SEED in $SEEDS; do
    # GCv1 Hungarian
    submit "movic_gcv1_pf_hung_v2_s${SEED}"  "$GC_MC_CFG" "$SN_HUNG_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_hung_v2_s${SEED}"  "$GC_MD_CFG" "$SN_HUNG_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_hung_v2_s${SEED}"  "$GC_ME_CFG" "$SN_HUNG_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_hung_v2_s${SEED}"  "$GC_YT_CFG" "$SN_HUNG_GC" "$SEED" "model.initializer.init_mode=per_frame"
    # GCv1 noST
    submit "movic_gcv1_pf_noST_v2_s${SEED}"  "$GC_MC_CFG" "$SN_NOST_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_noST_v2_s${SEED}"  "$GC_MD_CFG" "$SN_NOST_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_noST_v2_s${SEED}"  "$GC_ME_CFG" "$SN_NOST_GC" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_noST_v2_s${SEED}"  "$GC_YT_CFG" "$SN_NOST_GC" "$SEED" "model.initializer.init_mode=per_frame"
    # GCv1 nomatch
    submit "movic_gcv1_pf_nomatch_v2_s${SEED}"  "$GC_MC_CFG" "$SN_NOMATCH" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_nomatch_v2_s${SEED}"  "$GC_MD_CFG" "$SN_NOMATCH" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_nomatch_v2_s${SEED}"  "$GC_ME_CFG" "$SN_NOMATCH" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_nomatch_v2_s${SEED}"  "$GC_YT_CFG" "$SN_NOMATCH" "$SEED" "model.initializer.init_mode=per_frame"

    # SC Hungarian (NEW — not in v1 100K grid)
    submit "movic_sc_hung_v2_s${SEED}" "$SC_MC_CFG" "$SN_HUNG_SC" "$SEED"
    submit "movid_sc_hung_v2_s${SEED}" "$SC_MD_CFG" "$SN_HUNG_SC" "$SEED"
    submit "movie_sc_hung_v2_s${SEED}" "$SC_ME_CFG" "$SN_HUNG_SC" "$SEED"
    submit "ytvis_sc_hung_v2_s${SEED}" "$SC_YT_CFG" "$SN_HUNG_SC" "$SEED"
    # SC default predictor (TransformerEncoder)
    submit "movic_sc_default_v2_s${SEED}" "$SC_MC_CFG" "$SN_DEFAULT_PRED" "$SEED"
    submit "movid_sc_default_v2_s${SEED}" "$SC_MD_CFG" "$SN_DEFAULT_PRED" "$SEED"
    submit "movie_sc_default_v2_s${SEED}" "$SC_ME_CFG" "$SN_DEFAULT_PRED" "$SEED"
    submit "ytvis_sc_default_v2_s${SEED}" "$SC_YT_CFG" "$SN_DEFAULT_PRED" "$SEED"
    # SC noST
    submit "movic_sc_noST_v2_s${SEED}" "$SC_MC_CFG" "$SN_NOST_SC" "$SEED"
    submit "movid_sc_noST_v2_s${SEED}" "$SC_MD_CFG" "$SN_NOST_SC" "$SEED"
    submit "movie_sc_noST_v2_s${SEED}" "$SC_ME_CFG" "$SN_NOST_SC" "$SEED"
    submit "ytvis_sc_noST_v2_s${SEED}" "$SC_YT_CFG" "$SN_NOST_SC" "$SEED"
    # SC nomatch
    submit "movic_sc_nomatch_v2_s${SEED}" "$SC_MC_CFG" "$SN_NOMATCH" "$SEED"
    submit "movid_sc_nomatch_v2_s${SEED}" "$SC_MD_CFG" "$SN_NOMATCH" "$SEED"
    submit "movie_sc_nomatch_v2_s${SEED}" "$SC_ME_CFG" "$SN_NOMATCH" "$SEED"
    submit "ytvis_sc_nomatch_v2_s${SEED}" "$SC_YT_CFG" "$SN_NOMATCH" "$SEED"
  done
fi

# ----- WAVE 3: full-FT + featrec=0 cross-cell — 80 runs -----
if is_wave "wave3"; then
  echo "=== WAVE 3: Full-FT + featrec=0 cross-cell @ 100K, n=5 — 80 runs ==="
  for SEED in $SEEDS; do
    # full-FT (GC + SC × 4 datasets × n=5 = 40)
    submit "movic_gcv1_pf_fullft_v2_s${SEED}"  "$GC_MC_CFG" "$SN_FULLFT" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_fullft_v2_s${SEED}"  "$GC_MD_CFG" "$SN_FULLFT" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_fullft_v2_s${SEED}"  "$GC_ME_CFG" "$SN_FULLFT" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_fullft_v2_s${SEED}"  "$GC_YT_CFG" "$SN_FULLFT" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_sc_fullft_v2_s${SEED}" "$SC_MC_CFG" "$SN_FULLFT" "$SEED"
    submit "movid_sc_fullft_v2_s${SEED}" "$SC_MD_CFG" "$SN_FULLFT" "$SEED"
    submit "movie_sc_fullft_v2_s${SEED}" "$SC_ME_CFG" "$SN_FULLFT" "$SEED"
    submit "ytvis_sc_fullft_v2_s${SEED}" "$SC_YT_CFG" "$SN_FULLFT" "$SEED"
    # featrec=0 cross-cell (GC + SC × 4 datasets × n=5 = 40)
    submit "movic_gcv1_pf_featrec0_v2_s${SEED}"  "$GC_MC_CFG" "$SN_FEATREC0" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movid_gcv1_pf_featrec0_v2_s${SEED}"  "$GC_MD_CFG" "$SN_FEATREC0" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movie_gcv1_pf_featrec0_v2_s${SEED}"  "$GC_ME_CFG" "$SN_FEATREC0" "$SEED" "model.initializer.init_mode=per_frame"
    submit "ytvis_gcv1_pf_featrec0_v2_s${SEED}"  "$GC_YT_CFG" "$SN_FEATREC0" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_sc_featrec0_v2_s${SEED}" "$SC_MC_CFG" "$SN_FEATREC0" "$SEED"
    submit "movid_sc_featrec0_v2_s${SEED}" "$SC_MD_CFG" "$SN_FEATREC0" "$SEED"
    submit "movie_sc_featrec0_v2_s${SEED}" "$SC_ME_CFG" "$SN_FEATREC0" "$SEED"
    submit "ytvis_sc_featrec0_v2_s${SEED}" "$SC_YT_CFG" "$SN_FEATREC0" "$SEED"
  done
fi

# ----- WAVE 4: ablations + D1-light — 75 runs -----
# GC × MOVi-C ablations: featrec sweep + LoRA rank sweep + R3 step sweep × n=5
# D1-light: SC × {MOVi-D, MOVi-E} w/o slot-slot × {ST, default_pred, nomatch} × n=5
if is_wave "wave4"; then
  echo "=== WAVE 4: Ablations (GC × MOVi-C sweeps) + D1-light @ 100K, n=5 — 75 runs ==="
  for SEED in $SEEDS; do
    # featrec sweep (skip 1.5 = headline default)
    submit "movic_gcv1_pf_abl_fw0p5_v2_s${SEED}"  "$GC_MC_CFG" "$SN_ABL_FW0p5" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_gcv1_pf_abl_fw1p0_v2_s${SEED}"  "$GC_MC_CFG" "$SN_ABL_FW1p0" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_gcv1_pf_abl_fw2p0_v2_s${SEED}"  "$GC_MC_CFG" "$SN_ABL_FW2p0" "$SEED" "model.initializer.init_mode=per_frame"
    # LoRA rank sweep (skip r=8 = headline default)
    submit "movic_gcv1_pf_abl_lora_r4_v2_s${SEED}"   "$GC_MC_CFG" "$SN_ABL_LORA_R4"  "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_gcv1_pf_abl_lora_r16_v2_s${SEED}"  "$GC_MC_CFG" "$SN_ABL_LORA_R16" "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_gcv1_pf_abl_lora_r32_v2_s${SEED}"  "$GC_MC_CFG" "$SN_ABL_LORA_R32" "$SEED" "model.initializer.init_mode=per_frame"
    # R3 step sweep (skip 5000 = headline default)
    submit "movic_gcv1_pf_abl_r3_0_v2_s${SEED}"      "$GC_MC_CFG" "$SN_ABL_R3_0"     "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_gcv1_pf_abl_r3_2500_v2_s${SEED}"   "$GC_MC_CFG" "$SN_ABL_R3_2500"  "$SEED" "model.initializer.init_mode=per_frame"
    submit "movic_gcv1_pf_abl_r3_10k_v2_s${SEED}"    "$GC_MC_CFG" "$SN_ABL_R3_10K"   "$SEED" "model.initializer.init_mode=per_frame"

    # D1-light: SC w/o slot-slot loss × {MOVi-D, MOVi-E} × {ST, default_pred, nomatch}
    submit "movid_sc_d1_st_v2_s${SEED}"      "$SC_MD_CFG" "$SN_D1_ST"      "$SEED"
    submit "movid_sc_d1_default_v2_s${SEED}" "$SC_MD_CFG" "$SN_D1_DEFAULT" "$SEED"
    submit "movid_sc_d1_nomatch_v2_s${SEED}" "$SC_MD_CFG" "$SN_D1_NOMATCH" "$SEED"
    submit "movie_sc_d1_st_v2_s${SEED}"      "$SC_ME_CFG" "$SN_D1_ST"      "$SEED"
    submit "movie_sc_d1_default_v2_s${SEED}" "$SC_ME_CFG" "$SN_D1_DEFAULT" "$SEED"
    submit "movie_sc_d1_nomatch_v2_s${SEED}" "$SC_ME_CFG" "$SN_D1_NOMATCH" "$SEED"
  done
fi

# Help
if ! is_wave "wave1" && ! is_wave "wave2" && ! is_wave "wave3" && ! is_wave "wave4" && ! is_wave "smoke"; then
  echo "Usage: bash $(basename $0) {smoke|wave1|wave2|wave3|wave4|all} [--live]"
  echo "Total: 1 smoke + 80 wave1 + 140 wave2 + 80 wave3 + 75 wave4 = 375 100K runs (+ 1 smoke)"
  echo "  smoke: 1 quick gpu-debug job (~5 min) to verify ckpt saving + new save folder"
  echo "  wave1: anchors (frozen + ST) × 8 cells × n=5 — 80 runs"
  echo "  wave2: predictor counterfactuals — 140 runs"
  echo "  wave3: full-FT + featrec=0 cross-cell — 80 runs"
  echo "  wave4: GC × MOVi-C sweeps + D1-light — 75 runs"
  echo ""
  echo "WITHOUT --live the script is in DRY-RUN MODE and will not call sbatch."
  echo ""
  echo "Save root: $OUTPUT_DIR"
  echo "Slurm logs: $SLURM_LOG_DIR"
  echo "experiment_group: $EXP_GROUP"
fi
