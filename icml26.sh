#!/bin/bash

# ytvis2021.yaml, movi_d.yaml, movi_e.yaml
CONFIG_FILE="configs/slotcontrast/movi_d.yaml"


CONFIGS=(

    # ytvis2021
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 42 2 3 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 43 2 3 local_consistency false 0.5"
    "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 2 44 2 3 local_consistency false 0.5"

    # # movi_d
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 0.5"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 0.5"


    # # movi_e
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 42 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 43 2 3 local_consistency false 1.0"
    # "per_frame_greedy_local_hungarian_w_contrast GreedyFeatureInit per_frame networks.HungarianPredictor 1 44 2 3 local_consistency false 1.0"
)
    
for config in "${CONFIGS[@]}"; do
    read -r exp_name init_name init_mode predictor neighbor_radius SEED n_iters f_n_iters saliency_mode skip_predictor saliency_alpha<<< "$config"

    echo "Submitting: $exp_name with config: $CONFIG_FILE"
    
    sbatch --job-name="sc_${exp_name}" triton_slurm.sh "${CONFIG_FILE}" \
        "experiment_name=slotcontrast_${exp_name}" \
        "model.initializer.name=${init_name}" \
        "model.initializer.init_mode=${init_mode}" \
        "model.initializer.neighbor_radius=${neighbor_radius}" \
        "model.initializer.saliency_mode=${saliency_mode}" \
        "model.initializer.saliency_alpha=${saliency_alpha}" \
        "model.predictor.name=${predictor}" \
        "model.grouper.n_iters=${n_iters}" \
        "model.latent_processor.first_step_corrector_args.n_iters=${f_n_iters}" \
        "model.latent_processor.skip_predictor=${skip_predictor}" \
        "seed=${SEED}"
done

echo "All jobs submitted!"
