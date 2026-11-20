# Lesson 11: Vision Transformers

## Prerequisites

Use the course `.venv` with PyTorch, NumPy, Pillow, and Matplotlib installed. These packages are installed for the hands-on examples in Stage 1. No dataset download or pretrained model is needed.

## Run

From the repository root:

```bash
uv run python examples/03-modern-computer-vision/lesson-11-vision-transformers/patch_attention.py
```

## Expected outputs

The script prints the image, patch-token, and attention shapes. It saves and displays `outputs/patch_attention.png`, showing the synthetic image, its patch grid, one token's attention weights, and the weighted color summary.

## Limitations

This is a tiny attention illustration, not a trained ViT. It uses each patch's average RGB as a toy embedding and similarity as attention; real ViTs learn patch projections and query/key/value weights from data.
