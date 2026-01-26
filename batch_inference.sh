#!/bin/bash

# Batch inference with metrics for YTVIS2021 validation videos

# # baseline inference
# python data/batch_inference.py \
#     --checkpoint /checkpoints/baseline_ytvis/checkpoints/step=100000-v1.ckpt \
#     --config configs/inference/ytvis2021_baseline.yaml \
#     --data-dir /data/ytvis2021_raw/valid \
#     --output-dir /data/ytvis2021_inference \
#     --n-slots 7 \
#     --device cuda

# grounded_correspondence inference
python data/batch_inference.py \
    --checkpoint /checkpoints/GC_ytvis/checkpoints/step=100000-v1.ckpt \
    --config configs/inference/ytvis2021_gc.yaml \
    --data-dir /data/ytvis2021_raw/valid \
    --output-dir /data/ytvis2021_inference \
    --n-slots 7 \
    --device cuda

# To process specific videos for testing:
# python data/batch_inference.py --checkpoint <path> --config <path> --video-ids 00f88c4f0a 01c88b5b60

# To limit number of videos:
# python data/batch_inference.py --checkpoint <path> --config <path> --max-videos 10
