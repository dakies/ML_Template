#!/bin/bash
#SBATCH --job-name=ml-sweep
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --array=0-9%4

# ============================================================
# Hyperparameter sweep using SLURM job arrays
# Runs 10 jobs (indices 0-9) with max 4 concurrent
# ============================================================
# Usage:
#   sbatch scripts/slurm/sweep.sbatch
# ============================================================

set -e

echo "============================================"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
mkdir -p logs/slurm

# Define hyperparameter grid
LEARNING_RATES=(1e-4 3e-4 1e-3 3e-3 1e-2)
WEIGHT_DECAYS=(0.01 0.05)

# Calculate indices
NUM_LR=${#LEARNING_RATES[@]}
LR_IDX=$((SLURM_ARRAY_TASK_ID % NUM_LR))
WD_IDX=$((SLURM_ARRAY_TASK_ID / NUM_LR))

LR=${LEARNING_RATES[$LR_IDX]}
WD=${WEIGHT_DECAYS[$WD_IDX]}

echo "Hyperparameters: lr=$LR, weight_decay=$WD"

python -m ml_template.train \
    model.learning_rate=$LR \
    model.weight_decay=$WD \
    experiment_name=sweep_${SLURM_ARRAY_JOB_ID} \
    hydra.run.dir=outputs/sweep/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

echo "============================================"
echo "End time: $(date)"
echo "============================================"

