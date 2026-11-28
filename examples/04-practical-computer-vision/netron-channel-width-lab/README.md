# Netron Lab 2 — Channel Width

Reduce a CNN convolution from 64 output channels to 32, inspect how the graph and classifier shapes change, and compare theoretical cost with CPU latency.

## Prerequisites

Use the repository `.venv` with PyTorch 2.6 or newer. Install ONNX, ONNX Runtime CPU, and Netron from the repository root:

```bash
uv pip install --python .venv/bin/python onnx onnxruntime netron
```

No dataset or pretrained weights are required. Both networks have random weights, so their predictions do not measure accuracy.

## Run

From the repository root:

```bash
uv run python examples/04-practical-computer-vision/netron-channel-width-lab/export_models.py
uv run netron examples/04-practical-computer-vision/netron-channel-width-lab/outputs/baseline.onnx
uv run netron examples/04-practical-computer-vision/netron-channel-width-lab/outputs/narrow.onnx
uv run python examples/04-practical-computer-vision/netron-channel-width-lab/inspect_onnx.py
uv run python examples/04-practical-computer-vision/netron-channel-width-lab/benchmark_onnx.py
```

Netron starts a local viewer. Keep the process open while inspecting the graph; use another terminal to open the second model.

## Expected outputs

`outputs/` contains `baseline.onnx` and `narrow.onnx`. The exporter checks both graph files, output shape, total parameters, and target MAC counts. The inspection script prints convolution input/output shapes, weight shapes, and group counts. The benchmark prints session startup, median single-image CPU latency, and file size.

## Limitations

The narrower network changes model capacity and is not functionally equivalent to the baseline. Train both models with the same dataset, split, and recipe before comparing accuracy. MAC counts do not predict latency exactly, and CPU timing does not predict RK3566 NPU performance.
