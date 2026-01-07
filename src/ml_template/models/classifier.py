"""Image classification model implementations."""

from typing import Any

import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

from ml_template.models.base import BaseModule


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration purposes.

    A lightweight CNN suitable for MNIST/CIFAR10.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        return self.classifier(x)


class ImageClassifier(BaseModule):
    """Lightning module for image classification.

    Wraps a backbone model with training/validation/test logic
    and metric computation.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        backbone: Backbone architecture ('simple_cnn').
        **kwargs: Additional arguments for BaseModule.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        backbone: str = "simple_cnn",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Build model
        if backbone == "simple_cnn":
            self.model = SimpleCNN(
                in_channels=in_channels,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier."""
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared logic for training/validation/test steps."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Log metrics
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        return loss, preds, y

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Training step."""
        loss, preds, y = self._shared_step(batch, "train")
        self.train_acc(preds, y)
        self.log("train/acc", self.train_acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Validation step."""
        loss, preds, y = self._shared_step(batch, "val")
        self.val_acc(preds, y)
        self.log("val/acc", self.val_acc, prog_bar=True, sync_dist=True)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Test step."""
        loss, preds, y = self._shared_step(batch, "test")
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test/acc", self.test_acc, sync_dist=True)
        self.log("test/f1", self.test_f1, sync_dist=True)
