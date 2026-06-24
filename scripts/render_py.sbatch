#!/bin/bash
#SBATCH --job-name=qrender
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/elec/t41020_egovla/slotcontrast_v2/v1_100k/slurm_logs/%x_%j.err
set -eo pipefail
export WRKDIR=/scratch/work/$USER
module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=/$WRKDIR/.huggingface_cache
export CONDA_PKGS_DIRS=/$WRKDIR/.conda_pkgs
export CONDA_ENVS_PATH=/$WRKDIR/.conda_envs
export TORCH_HOME=/$WRKDIR/.cache/torch
source activate slotcontrast
export PYTHONPATH="${PWD}:$PYTHONPATH"
cd /scratch/work/$USER/slotcontrast
nvidia-smi --query-gpu=name --format=csv,noheader
python3 "$@"
date
