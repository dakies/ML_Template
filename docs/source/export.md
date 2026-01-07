# Model Export

Export trained models to ONNX format for deployment.

## ONNX Export

### Command Line

```bash
# Basic export
python -m ml_template.export checkpoints/best.ckpt -o model.onnx

# Custom input shape
python -m ml_template.export checkpoints/best.ckpt \
    --input-shape 1 3 224 224 \
    -o model.onnx

# With benchmarking
python -m ml_template.export checkpoints/best.ckpt --benchmark
```

### During Training

Enable automatic export after training:

```bash
python -m ml_template.train \
    export_onnx=true \
    export_onnx_path=outputs/model.onnx
```

Or use the full training experiment:

```bash
python -m ml_template.train experiment=full_train
```

### Programmatic Export

```python
from ml_template.models import ImageClassifier

# Load checkpoint
model = ImageClassifier.load_from_checkpoint("checkpoints/best.ckpt")

# Export to ONNX
model.export_to_onnx(
    "model.onnx",
    input_shape=(1, 3, 32, 32),
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

## Verify Export

```python
import onnx
import onnxruntime as ort
import numpy as np

# Load and check model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Run inference
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: np.random.randn(1, 3, 32, 32).astype(np.float32)})
print(f"Output shape: {output[0].shape}")
```

## Benchmark

```python
from ml_template.export import benchmark_onnx_model

results = benchmark_onnx_model(
    "model.onnx",
    input_shape=(1, 3, 32, 32),
    num_runs=100,
)
# Output: Mean: 1.23 ms, Throughput: 813.0 FPS
```

## Deployment Options

### ONNX Runtime (CPU/GPU)

```python
import onnxruntime as ort

# CPU
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# GPU
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
```

### TensorRT

```bash
# Convert to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt
```

### OpenVINO

```bash
mo --input_model model.onnx --output_dir openvino_model
```

