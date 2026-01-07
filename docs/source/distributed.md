# Distributed Training

This template supports multiple distributed training strategies.

## Single Node Multi-GPU (DDP)

```bash
# Auto-detect all GPUs
python -m ml_template.train trainer=ddp

# Specify GPU count
python -m ml_template.train trainer=ddp trainer.devices=4
```

## DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) enables efficient large-scale training.

```bash
# ZeRO Stage 2
python -m ml_template.train trainer=deepspeed
```

DeepSpeed stages:
- **Stage 1**: Optimizer state partitioning
- **Stage 2**: Gradient + optimizer partitioning
- **Stage 3**: Full parameter partitioning

## SLURM Clusters

### Single Node Job

```bash
sbatch scripts/slurm/train.sbatch
```

### Multi-Node Distributed

```bash
sbatch scripts/slurm/distributed.sbatch
```

### Hyperparameter Sweep

```bash
# Submits 10 jobs with different hyperparameters
sbatch scripts/slurm/sweep.sbatch
```

### Custom SLURM Job

```bash
sbatch scripts/slurm/train.sbatch \
    experiment=full_train \
    model.learning_rate=0.001
```

## Environment Variables

For multi-node training, these are set automatically by SLURM:

```bash
export MASTER_ADDR=<master_node>
export MASTER_PORT=29500
export WORLD_SIZE=<total_gpus>
export NODE_RANK=<node_index>
```

## Scaling Tips

1. **Linear scaling rule**: When increasing batch size by N, scale learning rate by N
2. **Gradient accumulation**: Use for memory-constrained setups
3. **Mixed precision**: Always enable for multi-GPU training
4. **NCCL tuning**: Set `NCCL_DEBUG=INFO` for debugging

## Docker with Multiple GPUs

```bash
# Run with all GPUs
docker run --gpus all ml-template \
    python -m ml_template.train trainer=ddp

# Specify GPUs
docker run --gpus '"device=0,1"' ml-template \
    python -m ml_template.train trainer.devices=2
```

