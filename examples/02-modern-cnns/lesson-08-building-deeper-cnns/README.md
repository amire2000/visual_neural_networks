# Lesson 8: Building Deeper CNNs

## Prerequisites

Use the course `.venv` with PyTorch and Matplotlib installed. This small architecture demo uses a random tensor, so it downloads no dataset or model weights.

## Run

From the repository root:

```bash
uv run python examples/02-modern-cnns/lesson-08-building-deeper-cnns/deeper_cnn_demo.py
```

## Expected output

The script prints tensor shapes after each pooling block and saves `outputs/deeper_cnn_comparison.png`, comparing classifier parameter counts before and after global average pooling.

## Limitations

The random input and weights demonstrate shape flow and layer structure, not learned park-camera predictions or accuracy.
