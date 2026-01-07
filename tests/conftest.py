"""Pytest fixtures and configuration."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sample batch for testing."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture
def sample_batch_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sample MNIST batch for testing."""
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    return images, labels
