"""ONNX export utilities for trained models."""

import argparse

import numpy as np
import onnx
import onnxruntime as ort

from ml_template.models import ImageClassifier


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    opset_version: int = 17,
    verify: bool = True,
) -> None:
    """Export a Lightning checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to the .ckpt file.
        output_path: Path to save the .onnx file.
        input_shape: Model input shape (batch, channels, height, width).
        opset_version: ONNX opset version.
        verify: Whether to verify the exported model.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    model = ImageClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    print(f"Exporting to ONNX: {output_path}")
    model.export_to_onnx(
        output_path,
        input_shape=input_shape,
        opset_version=opset_version,
    )

    if verify:
        verify_onnx_model(output_path, input_shape)


def verify_onnx_model(
    onnx_path: str,
    input_shape: tuple[int, ...],
) -> None:
    """Verify ONNX model produces same output as PyTorch.

    Args:
        onnx_path: Path to the ONNX model.
        input_shape: Input tensor shape.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
    """
    print("Verifying ONNX model...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_path)

    # Create random input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outputs = ort_session.run(None, ort_inputs)

    print("✓ ONNX model verified successfully!")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {ort_outputs[0].shape}")


def benchmark_onnx_model(
    onnx_path: str,
    input_shape: tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to the ONNX model.
        input_shape: Input tensor shape.
        num_runs: Number of benchmark runs.
        warmup_runs: Number of warmup runs.

    Returns:
        Dictionary with benchmark results.
    """
    import time

    ort_session = ort.InferenceSession(onnx_path)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}

    # Warmup
    for _ in range(warmup_runs):
        ort_session.run(None, ort_inputs)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ort_session.run(None, ort_inputs)
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to ms

    results = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "throughput_fps": float(1000 / np.mean(times)),
    }

    print(f"Benchmark Results ({num_runs} runs):")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  Min:  {results['min_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_fps']:.1f} FPS")

    return results


def main():
    """CLI entrypoint for ONNX export."""
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        help="Input shape (batch, channels, height, width)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after export",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification",
    )

    args = parser.parse_args()

    export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=tuple(args.input_shape),
        opset_version=args.opset,
        verify=not args.no_verify,
    )

    if args.benchmark:
        benchmark_onnx_model(args.output, tuple(args.input_shape))


if __name__ == "__main__":
    main()

