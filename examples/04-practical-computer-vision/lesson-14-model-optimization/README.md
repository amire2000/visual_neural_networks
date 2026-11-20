# Lesson 14: Model Optimization

## Prerequisites

Use the course `.venv` with PyTorch, TorchVision, Pillow, and Matplotlib installed. The script uses a randomly initialized MobileNetV3-Small as an inference workload; it downloads no weights or data. A CUDA GPU is needed to compare GPU FP32 and FP16 timings. CPU-only systems still run the benchmark.

## Run

From the repository root:

```bash
uv run python examples/04-practical-computer-vision/lesson-14-model-optimization/optimization_benchmark.py
```

## Expected outputs

The script reports median single-image inference latency for available devices and saves/displays `outputs/optimization_tradeoffs.png`. The figure compares estimated parameter storage for FP32, FP16, and INT8 with measured FP32/FP16 latency and a sample conveyor-camera image.

## Limitations

Weights are random, so this measures inference speed and numerical differences, not classification quality. INT8 storage is an estimate; this example does not run an INT8 kernel or TensorRT. Latency depends on hardware and background load.
