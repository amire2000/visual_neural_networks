# MNIST Vision Transformer

Train a compact patch-based Vision Transformer on MNIST, then run PyTorch inference on a random batch from the official test split.

## Prerequisites

Complete [Setup Lab](../../docs/lesson-00-setup-lab/index.md) and install PyTorch, TorchVision, and Matplotlib into the course `.venv`. Run commands from the repository root. The scripts download MNIST to `data/mnist/` if it is missing.

## Run the project

```bash
uv run python examples/capstone-mnist-vit/train_vit_mnist.py
uv run python examples/capstone-mnist-vit/infer_vit_mnist.py
```

Training uses CUDA when the installed PyTorch build supports it, otherwise CPU. Inference requires the checkpoint produced by training.

## Expected outputs

The example's `outputs/` directory contains:

- `best_vit.pth` — weights from the epoch with best validation accuracy.
- `training_curves.png` — training and validation loss and accuracy.
- `vit_predictions.png` — a 4 × 4 visual grid of true labels and predictions.

The inference script also prints accuracy for the displayed 16-image batch. That sample is not full test-set accuracy.

## Limitations

This is a small teaching ViT trained from scratch on a simple dataset. It is useful for learning patches, embeddings, a class token, encoder layers, training, and inference. It is not a pretrained production model, a state-of-the-art MNIST benchmark, or evidence that ViTs outperform CNNs on this task. Training duration and accuracy depend on the device.
