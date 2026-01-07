#!/bin/bash
#SBATCH --job-name=ml-template
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# ============================================================
# Single-node GPU training script for SLURM clusters
# ============================================================
# Usage:
#   sbatch scripts/slurm/train.sbatch
#   sbatch scripts/slurm/train.sbatch experiment=full_train
# ============================================================

set -e

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

# Load required modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.11

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source .venv/bin/activate

# Create log directory
mkdir -p logs/slurm

# Run training with any additional arguments passed to sbatch
python -m ml_template.train \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    hydra.run.dir=outputs/slurm/${SLURM_JOB_ID} \
    "$@"

echo "============================================"
echo "End time: $(date)"
echo "============================================"

