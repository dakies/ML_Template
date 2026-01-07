"""Tests for model implementations."""

import torch

from ml_template.models import ImageClassifier
from ml_template.models.classifier import SimpleCNN


class TestSimpleCNN:
    """Tests for SimpleCNN architecture."""

    def test_forward_cifar(self) -> None:
        """Test forward pass with CIFAR-10 input shape."""
        model = SimpleCNN(in_channels=3, num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_mnist(self) -> None:
        """Test forward pass with MNIST input shape."""
        model = SimpleCNN(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_different_num_classes(self) -> None:
        """Test model with different number of classes."""
        model = SimpleCNN(in_channels=3, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)


class TestImageClassifier:
    """Tests for ImageClassifier Lightning module."""

    def test_forward(self, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test forward pass."""
        model = ImageClassifier(num_classes=10, in_channels=3)
        images, _ = sample_batch
        out = model(images)
        assert out.shape == (4, 10)

    def test_training_step(self, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test training step returns loss."""
        model = ImageClassifier(num_classes=10, in_channels=3)
        loss = model.training_step(sample_batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_validation_step(self, sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test validation step runs without error."""
        model = ImageClassifier(num_classes=10, in_channels=3)
        model.validation_step(sample_batch, batch_idx=0)

    def test_configure_optimizers(self) -> None:
        """Test optimizer configuration."""
        model = ImageClassifier(learning_rate=1e-3, scheduler="cosine")
        # Mock trainer for scheduler configuration
        model.trainer = type(
            "Trainer",
            (),
            {
                "estimated_stepping_batches": 1000,
                "max_epochs": 10,
            },
        )()
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

    def test_no_scheduler(self) -> None:
        """Test configuration without scheduler."""
        model = ImageClassifier(scheduler=None)
        model.trainer = None
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" not in opt_config
