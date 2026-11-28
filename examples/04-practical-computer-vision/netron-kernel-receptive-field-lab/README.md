# Netron Lab 3 — Receptive Field and Kernel Size

Compare one 5×5 convolution with two stacked 3×3 convolutions. Inspect the ONNX graphs, calculate MACs, and measure CPU inference.

## Prerequisites

Use the repository `.venv` with PyTorch 2.6 or newer. Install ONNX, ONNX Runtime CPU, and Netron from the repository root:

```bash
uv pip install --python .venv/bin/python onnx onnxruntime netron
```

No dataset or pretrained weights are required. The models start with random weights, so their classification outputs do not indicate accuracy.

## Run

From the repository root:

```bash
uv run python examples/04-practical-computer-vision/netron-kernel-receptive-field-lab/models.py
uv run netron examples/04-practical-computer-vision/netron-kernel-receptive-field-lab/outputs/baseline.onnx
uv run netron examples/04-practical-computer-vision/netron-kernel-receptive-field-lab/outputs/optimized.onnx
uv run python examples/04-practical-computer-vision/netron-kernel-receptive-field-lab/inspect_onnx.py
uv run python examples/04-practical-computer-vision/netron-kernel-receptive-field-lab/benchmark_onnx.py
```

Netron starts a local viewer. Keep it running while you inspect a graph; use another terminal for the other model.

## Expected outputs

`outputs/` contains `baseline.onnx` and `optimized.onnx`. The model script validates each ONNX graph, output shape, expected parameter count, and target MAC count. The inspector reports convolution weight shapes and MACs. The benchmark prints session startup, median single-image CPU latency, and model size.

## Optional RK3566 conversion

On an x86 Linux host with a compatible RKNN-Toolkit2 installation, run from the repository root:

```bash
uv run python examples/04-practical-computer-vision/netron-kernel-receptive-field-lab/convert_rknn.py
```

Benchmark the exported `.rknn` models on the actual RK3566 device. Host CPU timing and Netron inspection do not predict NPU latency or profiling results.

## Limitations

Two 3×3 layers have a 5×5 theoretical receptive field at stride 1, but the extra ReLU makes the computation different from one 5×5 layer. The models are untrained; train both with the same data and procedure before comparing accuracy. Fewer MACs do not guarantee lower latency on every device.
