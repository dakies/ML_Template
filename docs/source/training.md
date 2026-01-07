# Training

## Basic Training

```bash
# Train with default settings (CIFAR-10, 20 epochs)
python -m ml_template.train

# Train on MNIST
python -m ml_template.train data=mnist model.in_channels=1

# Quick debug run
python -m ml_template.train experiment=debug
```

## Monitoring

### TensorBoard

TensorBoard logging is enabled by default:

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# Open http://localhost:6006
```

### MLflow

Enable MLflow tracking:

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Train with MLflow
python -m ml_template.train \
    logger.mlflow.enabled=true \
    logger.mlflow.tracking_uri=http://localhost:5000
```

Or use Docker Compose:

```bash
cd docker
docker-compose up
```

## Checkpointing

Checkpoints are saved automatically:

```yaml
# configs/trainer/default.yaml
checkpoint_dir: checkpoints
save_top_k: 3        # Keep top 3 by val/acc
```

Resume from checkpoint:

```bash
python -m ml_template.train \
    trainer.ckpt_path=checkpoints/last.ckpt
```

## Early Stopping

Enabled by default, configure in trainer config:

```yaml
early_stopping: true
patience: 5          # Stop after 5 epochs without improvement
```

## Mixed Precision

Speed up training with mixed precision:

```bash
python -m ml_template.train trainer.precision=16-mixed
```

Options:
- `32-true`: Full precision (default)
- `16-mixed`: FP16 mixed precision
- `bf16-mixed`: BF16 mixed precision (Ampere+ GPUs)

## Gradient Accumulation

For effective larger batch sizes:

```bash
# Effective batch = 64 * 4 = 256
python -m ml_template.train \
    data.batch_size=64 \
    trainer.accumulate_grad_batches=4
```

