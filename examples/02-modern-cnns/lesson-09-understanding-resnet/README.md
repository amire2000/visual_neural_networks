# Lesson 9: Understanding ResNet

## Prerequisites

Use the course `.venv` with PyTorch and Matplotlib installed. The example uses a tiny feature map and downloads no data or model weights.

## Run

From the repository root:

```bash
uv run python examples/02-modern-cnns/lesson-09-understanding-resnet/residual_demo.py
```

## Expected output

The script prints the input, change, and output shapes, then saves `outputs/residual_addition.png`. The three panels use one shared color scale so the element-wise addition remains easy to compare.

## Limitations

The values are a hand-built teaching example, not activations from a trained ResNet.
