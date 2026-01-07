"""ONNX export callback for automatic model export."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback


class ONNXExportCallback(Callback):
    """Callback to export model to ONNX format after training.

    Automatically exports the best model checkpoint to ONNX format
    at the end of training.

    Args:
        export_path: Path to save the ONNX model.
        input_shape: Shape of input tensor (including batch dim).
        opset_version: ONNX opset version.
        export_on_train_end: Whether to export at end of training.
    """

    def __init__(
        self,
        export_path: str = "model.onnx",
        input_shape: tuple[int, ...] = (1, 3, 32, 32),
        opset_version: int = 17,
        export_on_train_end: bool = True,
    ) -> None:
        super().__init__()
        self.export_path = Path(export_path)
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.export_on_train_end = export_on_train_end

    def on_train_end(
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: L.LightningModule,  # noqa: ARG002
    ) -> None:
        """Export model to ONNX at end of training."""
        if not self.export_on_train_end:
            return

        # Ensure export directory exists
        self.export_path.parent.mkdir(parents=True, exist_ok=True)

        # Export using the model's built-in method if available
        export_method = getattr(pl_module, "export_to_onnx", None)
        if export_method is not None and callable(export_method):
            export_method(
                str(self.export_path),
                input_shape=self.input_shape,
                opset_version=self.opset_version,
            )
        else:
            # Fallback to basic export
            import torch

            pl_module.eval()
            dummy_input = torch.randn(*self.input_shape, device=pl_module.device)
            torch.onnx.export(
                pl_module,
                (dummy_input,),
                str(self.export_path),
                opset_version=self.opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            print(f"✓ ONNX model exported: {self.export_path}")
