# Netron Network Optimization Lab

Export two small CNNs to ONNX, inspect their operators and tensor shapes in Netron, then compare theoretical target-layer MACs with measured CPU inference.

## Prerequisites

Use the repository `.venv` with PyTorch 2.6 or newer. Install ONNX, ONNX Runtime CPU, and Netron from the repository root:

```bash
uv pip install --python .venv/bin/python onnx onnxruntime netron
```

No dataset or pretrained weights are required. The networks are randomly initialized, so predictions and accuracy are not meaningful.

## Run

From the repository root:

```bash
uv run python examples/04-practical-computer-vision/netron-network-optimization-lab/export_models.py
uv run netron examples/04-practical-computer-vision/netron-network-optimization-lab/outputs/original.onnx
uv run netron examples/04-practical-computer-vision/netron-network-optimization-lab/outputs/optimized.onnx
uv run python examples/04-practical-computer-vision/netron-network-optimization-lab/benchmark_onnx.py
```

Netron starts a local viewer; keep each server process open while viewing its graph. Run the next command in another terminal if you want both graphs open at once.

## Expected outputs

`outputs/` contains `original.onnx` and `optimized.onnx`. The export script checks each ONNX graph, prints total parameters and target-layer weight/MAC estimates, and asserts expected output shapes and MAC counts. Netron displays the model graphs and tensor metadata. The benchmark prints session startup, median single-image CPU latency, and ONNX file size for each graph.

## Limitations

The depthwise-separable target layer has fewer theoretical MACs and weights, but it is not mathematically equivalent to the standard convolution it replaces. This demo does not train either model, so it cannot compare classification quality. CPU latency is hardware- and runtime-dependent; Netron visualizes graphs but does not profile layer execution. The optional RK3566/RKNN workflow requires separate vendor software and an RK3566 device; see the lab page for boundaries and follow-up work.
