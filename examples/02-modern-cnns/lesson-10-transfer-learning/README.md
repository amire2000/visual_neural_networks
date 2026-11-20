# Lesson 10: Transfer Learning

## Prerequisites

Use the course `.venv` with PyTorch, TorchVision, and Matplotlib installed. The first run downloads TorchVision's pretrained ResNet-18 weights (about 45 MB); later runs reuse the local cache.

## Run

From the repository root:

```bash
uv run python examples/02-modern-cnns/lesson-10-transfer-learning/transfer_learning_demo.py
```

## Expected output

The script reports the ten-logit output shape and the number of trainable parameters for a new classifier head. It saves `outputs/transfer_learning_comparison.png`, which compares head-only training with unfreezing the final ResNet stage and the head.

## Limitations

The script demonstrates model adaptation and parameter selection; it does not train on CIFAR-10 or report accuracy. Input is random so no dataset download is needed.
