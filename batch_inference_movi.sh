#!/bin/bash

# Batch inference on MOVI datasets (no GT masks, just visualizations)

# baseline
# MOVI-D inference
echo "Running inference on MOVI-D validation..."
# python data/batch_inference.py \
#     --checkpoint checkpoints/baseline_movid/checkpoints/step=100000-v1.ckpt \
#     --config configs/inference/movi_d_baseline.yaml \
#     --data-dir data/movi_d_raw/valid \
#     --output-dir data/movi_inference \
#     --n-slots 15 \
#     --device cuda

# # MOVI-E inference
# echo "Running inference on MOVI-E validation..."
# python data/batch_inference.py \
#     --checkpoint checkpoints/baseline_movie/checkpoints/step=100000-v1.ckpt \
#     --config configs/inference/movi_e_baseline.yaml \
#     --data-dir data/movi_e_raw/valid \
#     --output-dir data/movi_inference \
#     --n-slots 15 \
#     --device cuda

# grounded_correspondence
# MOVI-D inference
python data/batch_inference.py \
    --checkpoint checkpoints/GC_movid/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/movi_d_gc.yaml \
    --data-dir data/movi_d_raw/valid \
    --output-dir data/movi_inference \
    --n-slots 15 \
    --device cuda

# MOVI-E inference
echo "Running inference on MOVI-E validation..."
python data/batch_inference.py \
    --checkpoint checkpoints/GC_movie/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/movi_e_gc.yaml \
    --data-dir data/movi_e_raw/valid \
    --output-dir data/movi_inference \
    --n-slots 15 \
    --device cuda

# To test with limited videos:
# python data/batch_inference.py --checkpoint <path> --config configs/slotcontrast/movi_d.yaml --data-dir data/movi_d_raw/valid --n-slots 15 --max-videos 10
