#!/bin/bash
#SBATCH --job-name=ml-template-dist
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --exclusive

# ============================================================
# Multi-node distributed training script for SLURM clusters
# Uses PyTorch DDP with NCCL backend
# ============================================================
# Usage:
#   sbatch scripts/slurm/distributed.sbatch
#   sbatch scripts/slurm/distributed.sbatch trainer=deepspeed
# ============================================================

set -e

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "============================================"

# Load required modules (adjust for your cluster)
# module load cuda/12.1
# module load python/3.11
# module load nccl

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment
source .venv/bin/activate

# Create log directory
mkdir -p logs/slurm

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Calculate total GPUs
GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Total GPUs: $TOTAL_GPUS"

# Launch distributed training with srun
srun --ntasks-per-node=1 python -m ml_template.train \
    trainer.strategy=ddp \
    trainer.devices=$GPUS_PER_NODE \
    trainer.num_nodes=$SLURM_NNODES \
    trainer.accelerator=gpu \
    hydra.run.dir=outputs/slurm/${SLURM_JOB_ID} \
    "$@"

echo "============================================"
echo "End time: $(date)"
echo "============================================"

