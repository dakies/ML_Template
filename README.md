# ML Template

<p align="center">
  <strong>A production-ready machine learning project template</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#usage">Usage</a> •
  <a href="#starting-a-new-project">New Project</a>
</p>

---

A professional ML project template demonstrating best practices in software engineering for machine learning. Built with PyTorch Lightning, Hydra, MLflow, and more.

## Features

| Category | Tools | Description |
|----------|-------|-------------|
| **Training** | PyTorch Lightning | Clean training loops, callbacks, logging |
| **Configuration** | Hydra + OmegaConf | Hierarchical configs, CLI overrides, sweeps |
| **Experiment Tracking** | MLflow + TensorBoard | Metrics, artifacts, model registry |
| **Distributed** | DeepSpeed, DDP | Multi-GPU and multi-node training |
| **Deployment** | ONNX | Export models for production inference |
| **Infrastructure** | Docker, SLURM | Containerization, HPC cluster support |
| **Code Quality** | Ruff, Ty, MyPy, Pre-commit | Linting, formatting, type checking |
| **CI/CD** | GitHub Actions | Automated testing and releases |
| **Documentation** | Sphinx | Auto-generated API docs |

## Quick Start

### Installation

```bash
# Clone the template
git clone https://github.com/yourusername/ml-template.git
cd ml-template

# Install dependencies (requires uv)
make install-dev

# Or with pip
pip install -e ".[dev]"
```

### Training

```bash
# Train with default config (CIFAR-10, 20 epochs)
make train

# Quick debug run
make train-debug

# Full training with ONNX export
python -m ml_template.train experiment=full_train

# Override any parameter
python -m ml_template.train \
    model.learning_rate=0.001 \
    trainer.max_epochs=50 \
    data=mnist
```

### Export to ONNX

```bash
python -m ml_template.export checkpoints/best.ckpt -o model.onnx --benchmark
```

## Project Structure

```
ml-template/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── model/              # Model architectures
│   ├── data/               # Dataset configs
│   ├── trainer/            # Training strategies (default, ddp, deepspeed)
│   └── experiment/         # Complete experiment presets
├── src/ml_template/        # Source code
│   ├── data/               # DataModules and transforms
│   ├── models/             # Lightning modules and architectures
│   ├── callbacks/          # Custom callbacks (ONNX export, etc.)
│   ├── utils/              # Utilities
│   ├── train.py            # Training entrypoint
│   └── export.py           # ONNX export utilities
├── scripts/slurm/          # SLURM job scripts
├── docker/                 # Dockerfile and compose
├── tests/                  # Test suite
├── docs/                   # Sphinx documentation
├── Makefile               # Development commands
└── pyproject.toml         # Project configuration
```

## Usage

### Configuration with Hydra

Override config groups or individual values via CLI:

```bash
# Switch dataset
python -m ml_template.train data=mnist

# Use DeepSpeed for distributed training
python -m ml_template.train trainer=deepspeed

# Hyperparameter sweep
python -m ml_template.train --multirun \
    model.learning_rate=1e-4,3e-4,1e-3
```

### Distributed Training

```bash
# Multi-GPU with DDP
python -m ml_template.train trainer=ddp

# DeepSpeed ZeRO Stage 2
python -m ml_template.train trainer=deepspeed

# SLURM cluster
sbatch scripts/slurm/distributed.sbatch
```

### Experiment Tracking

```bash
# Start MLflow UI
mlflow server --host 0.0.0.0 --port 5000

# Train with MLflow logging
python -m ml_template.train \
    logger.mlflow.enabled=true \
    logger.mlflow.tracking_uri=http://localhost:5000
```

Or use Docker Compose:
```bash
cd docker && docker-compose up
```

### Docker

```bash
# Build image
make docker

# Run training
docker run --gpus all ml-template python -m ml_template.train

# Development container with MLflow
cd docker && docker-compose up
```

## Starting a New Project

Use this template as a starting point for your ML projects:

### 1. Create from Template

```bash
# Using GitHub template feature (recommended)
# Click "Use this template" on GitHub

# Or clone and reset git history
git clone https://github.com/yourusername/ml-template.git my-project
cd my-project
rm -rf .git && git init
```

### 2. Customize the Template

1. **Update project metadata** in `pyproject.toml`:
   ```toml
   [project]
   name = "your-project-name"
   description = "Your project description"
   ```

2. **Rename the package**:
   ```bash
   mv src/ml_template src/your_project
   # Update imports in all files
   ```

3. **Add your model** in `src/your_project/models/`:
   ```python
   from your_project.models.base import BaseModule

   class MyModel(BaseModule):
       def __init__(self, ...):
           super().__init__()
           self.model = ...  # Your architecture
   ```

4. **Add your dataset** in `src/your_project/data/`:
   ```python
   from your_project.data.datamodule import BaseDataModule

   class MyDataModule(BaseDataModule):
       def prepare_data(self):
           ...
       def setup(self, stage):
           ...
   ```

5. **Create configs** in `configs/`:
   ```yaml
   # configs/model/my_model.yaml
   _target_: your_project.models.MyModel
   param1: value1
   ```

### 3. Development Workflow

```bash
# Install dev dependencies
make install-dev

# Run linting and tests
make lint
make test

# Train and iterate
make train-debug
```

## Development

```bash
make help          # Show all commands
make lint          # Run linter and type checker
make format        # Format code
make test          # Run tests
make test-cov      # Tests with coverage
make docs          # Build documentation
```

## Configuration Reference

### Model (`configs/model/`)
- `num_classes`: Number of output classes
- `learning_rate`: Optimizer learning rate
- `scheduler`: LR scheduler (cosine, onecycle, null)

### Data (`configs/data/`)
- `dataset_name`: Dataset to use (cifar10, mnist)
- `batch_size`: Training batch size
- `num_workers`: DataLoader workers

### Trainer (`configs/trainer/`)
- `max_epochs`: Maximum training epochs
- `precision`: Training precision (32-true, 16-mixed, bf16-mixed)
- `strategy`: Distributed strategy (null, ddp, deepspeed_stage_2)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for the ML community
</p>

