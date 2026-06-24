#!/bin/bash
#SBATCH --job-name=slotcontrast_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

#--gres=gpu:h200_2g.35gb:1
#--partition=gpu-h200-35g-ia

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=/$WRKDIR/.huggingface_cache
export PIP_CACHE_DIR=/$WRKDIR/.pip_cache
export CONDA_PKGS_DIRS=/$WRKDIR/.conda_pkgs
export CONDA_ENVS_PATH=/$WRKDIR/.conda_envs
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions
export WANDB_DIR=/$WRKDIR/wandb
export WANDB_CACHE_DIR=/$WRKDIR/wandb_cache

source activate slotcontrast

export PYTHONPATH="${PWD}:$PYTHONPATH"

DATA_DIR="/scratch/work/liz23/slotcontrast/data"
# Default log root; overridable from the launcher via `export OUTPUT_DIR=...`
# before sbatch (sbatch propagates env by default).
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/work/liz23/slotcontrast/logs}"

# First argument is the config file, rest are additional arguments
CONFIG_FILE=$1
shift

# Avoid "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, SLURM_MEM_PER_NODE are mutually
# exclusive" when the override partition has a default mem-per-GPU that
# collides with the #SBATCH --mem=128G (which sets SLURM_MEM_PER_NODE).
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

srun --mem=128G python slotcontrast/train.py "${CONFIG_FILE}" \
        "$@" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR} \
