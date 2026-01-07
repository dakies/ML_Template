# Configuration

This project uses [Hydra](https://hydra.cc/) for hierarchical configuration management.

## Config Structure

```
configs/
├── config.yaml          # Main config (composes others)
├── model/
│   └── classifier.yaml  # Model architecture settings
├── data/
│   ├── cifar10.yaml     # CIFAR-10 dataset config
│   └── mnist.yaml       # MNIST dataset config
├── trainer/
│   ├── default.yaml     # Default trainer settings
│   ├── ddp.yaml         # DDP distributed training
│   └── deepspeed.yaml   # DeepSpeed configuration
└── experiment/
    ├── debug.yaml       # Quick debug preset
    └── full_train.yaml  # Full training preset
```

## Basic Usage

```bash
# Use defaults
python -m ml_template.train

# Override config group
python -m ml_template.train data=mnist trainer=deepspeed

# Override individual values
python -m ml_template.train model.learning_rate=0.001

# Use experiment preset
python -m ml_template.train experiment=full_train
```

## Creating Custom Configs

### Custom Model Config

Create `configs/model/my_model.yaml`:

```yaml
num_classes: 100
in_channels: 3
backbone: simple_cnn
learning_rate: 5e-4
weight_decay: 0.01
scheduler: onecycle
```

Use it:
```bash
python -m ml_template.train model=my_model
```

### Custom Experiment

Create `configs/experiment/my_experiment.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: classifier
  - override /data: cifar10
  - override /trainer: deepspeed

experiment_name: my_experiment

trainer:
  max_epochs: 200
  precision: 16-mixed

model:
  learning_rate: 3e-4
```

## Hyperparameter Sweeps

```bash
# Multirun with Hydra
python -m ml_template.train --multirun \
    model.learning_rate=1e-4,3e-4,1e-3 \
    model.weight_decay=0.01,0.05
```

## Environment Variables

Use environment variables in configs:

```yaml
data_dir: ${oc.env:DATA_DIR,./data}
```

## Output Directory

Hydra automatically creates output directories:

```
outputs/
└── experiment_name/
    └── 2024-01-15_10-30-00/
        ├── .hydra/           # Saved configs
        ├── checkpoints/      # Model checkpoints
        └── train.log         # Training log
```

