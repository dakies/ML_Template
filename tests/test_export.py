"""Tests for ONNX export functionality."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml_template.models import ImageClassifier


class TestONNXExport:
    """Tests for ONNX model export."""

    def test_export_to_onnx(self) -> None:
        """Test basic ONNX export."""
        model = ImageClassifier(num_classes=10, in_channels=3)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            model.export_to_onnx(
                str(onnx_path),
                input_shape=(1, 3, 32, 32),
            )

            # Verify file exists and is valid
            assert onnx_path.exists()
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

    def test_onnx_inference(self) -> None:
        """Test ONNX model produces output."""
        model = ImageClassifier(num_classes=10, in_channels=3)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            model.export_to_onnx(str(onnx_path), input_shape=(1, 3, 32, 32))

            # Run inference with ONNX Runtime
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)

            outputs = session.run(None, {input_name: dummy_input})

            assert len(outputs) == 1
            output = outputs[0]
            assert hasattr(output, "shape")
            assert output.shape == (1, 10)  # type: ignore[union-attr]

    def test_onnx_output_consistency(self) -> None:
        """Test ONNX model output matches PyTorch output."""
        model = ImageClassifier(num_classes=10, in_channels=3)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            model.export_to_onnx(str(onnx_path), input_shape=(1, 3, 32, 32))

            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name

            # Test with fixed input
            dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
            onnx_output = session.run(None, {input_name: dummy_input})[0]

            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = model(torch.from_numpy(dummy_input)).numpy()

            # Ensure both are numpy arrays
            onnx_array = np.asarray(onnx_output)
            torch_array = np.asarray(torch_output)
            np.testing.assert_allclose(onnx_array, torch_array, rtol=1e-3, atol=1e-5)
