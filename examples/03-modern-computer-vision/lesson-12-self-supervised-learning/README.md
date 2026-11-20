# Lesson 12: Self-Supervised Learning

## Prerequisites

Use the course `.venv` with PyTorch, Pillow, and Matplotlib installed. The demo creates its own tiny geometric image set and needs no download or class labels.

## Run

From the repository root:

```bash
uv run python examples/03-modern-computer-vision/lesson-12-self-supervised-learning/contrastive_views.py
```

## Expected outputs

The script reports the loss by epoch and saves and displays `outputs/contrastive_views.png`. The figure shows two transformed views of an unlabeled image, the loss curve, and view-similarity matrices before and after training.

## Limitations

This is a small SimCLR-style contrastive teaching demo, not DINO or a general-purpose vision model. It uses synthetic grayscale patterns and treats other images in each batch as negatives. DINO uses student–teacher predictions instead of this explicit negative-pair loss.
