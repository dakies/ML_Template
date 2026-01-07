# Installation

## Requirements

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- uv (recommended) or pip

## Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/ml-template.git
cd ml-template
uv sync
```

## Using pip

```bash
git clone https://github.com/yourusername/ml-template.git
cd ml-template
pip install -e ".[dev]"
```

## Using Docker

```bash
# Build the image
docker build -f docker/Dockerfile -t ml-template .

# Run training
docker run --gpus all ml-template python -m ml_template.train

# With docker-compose (includes MLflow)
cd docker
docker-compose up
```

## Development Setup

```bash
# Install with dev dependencies
uv sync --dev

# Setup pre-commit hooks
uv run pre-commit install

# Verify installation
uv run pytest tests -v
```

## GPU Support

For NVIDIA GPUs, ensure you have:

1. NVIDIA drivers installed
2. CUDA toolkit (12.1+)
3. cuDNN

Verify GPU is available:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

