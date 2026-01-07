# Quickstart

Get up and running in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-template.git
cd ml-template

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Run Training

```bash
# Basic training with default config
python -m ml_template.train

# Train with MNIST instead of CIFAR-10
python -m ml_template.train data=mnist

# Quick debug run
python -m ml_template.train experiment=debug

# Full training with ONNX export
python -m ml_template.train experiment=full_train
```

## Configuration

All settings are managed through Hydra configs in `configs/`:

```bash
# Override any parameter via CLI
python -m ml_template.train \
    model.learning_rate=0.001 \
    trainer.max_epochs=50 \
    data.batch_size=128
```

## Export Model

```bash
# Export trained model to ONNX
python -m ml_template.export checkpoints/best.ckpt -o model.onnx

# With benchmarking
python -m ml_template.export checkpoints/best.ckpt --benchmark
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for customization
- Learn about [Distributed Training](distributed.md) for multi-GPU setups
- Check the [API Reference](api/modules.rst) for detailed documentation

