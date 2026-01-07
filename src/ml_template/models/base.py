"""Base Lightning module with common training logic."""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


class BaseModule(L.LightningModule):
    """Base Lightning module with standardized training setup.

    Provides common functionality for:
    - Optimizer and scheduler configuration
    - Step/epoch logging
    - ONNX export support

    Attributes:
        learning_rate: Base learning rate for optimizer.
        weight_decay: Weight decay for regularization.
        scheduler: Learning rate scheduler type ('cosine', 'onecycle', or None).
        warmup_epochs: Number of warmup epochs for schedulers.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        scheduler: str | None = "cosine",
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.warmup_epochs = warmup_epochs

        # To be defined in subclass
        self.model: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_type is None:
            return {"optimizer": optimizer}

        # Calculate total steps for schedulers
        if self.trainer and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 10000  # Fallback

        max_epochs = self.trainer.max_epochs if self.trainer else 100

        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        elif self.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=self.warmup_epochs / max_epochs if max_epochs > 0 else 0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return {"optimizer": optimizer}

    def export_to_onnx(
        self,
        filepath: str,
        input_shape: tuple[int, ...],
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 17,
    ) -> None:
        """Export model to ONNX format.

        Args:
            filepath: Path to save the ONNX model.
            input_shape: Shape of input tensor (including batch dimension).
            dynamic_axes: Dynamic axes for variable-length dimensions.
            opset_version: ONNX opset version to use.

        Example:
            >>> model.export_to_onnx(
            ...     "model.onnx",
            ...     input_shape=(1, 3, 32, 32),
            ...     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            ... )
        """
        self.eval()
        dummy_input = torch.randn(*input_shape, device=self.device)

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        # Verify the exported model
        import onnx

        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model exported and verified: {filepath}")
