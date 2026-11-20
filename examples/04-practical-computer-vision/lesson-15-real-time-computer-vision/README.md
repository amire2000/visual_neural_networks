# Lesson 15: Real-Time Computer Vision

## Prerequisites

Use the course `.venv` with PyTorch, TorchVision, Pillow, and Matplotlib installed. The script uses a randomly initialized MobileNetV3-Small and generated camera frames; it downloads no data or weights. CUDA is used when available.

## Run

From the repository root:

```bash
uv run python examples/04-practical-computer-vision/lesson-15-real-time-computer-vision/streaming_benchmark.py
```

## Expected outputs

The script measures resize with the camera aspect ratio preserved, normalization, host-to-device transfer, inference, and class-score selection for 224 and 320 pixel short edges at several batch sizes. It saves/displays `outputs/streaming_tradeoffs.png` with estimated response time and measured throughput.

## Limitations

The model weights are random and only stand in for an image workload. Camera capture and display are excluded. Batch-fill wait is an average estimate for a steady 30 FPS stream. Timings vary by hardware and background load.
