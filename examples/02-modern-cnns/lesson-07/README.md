# Lesson 7: From MNIST to CIFAR-10

## Prerequisites

Use the course `.venv` with PyTorch, TorchVision, and Matplotlib installed. If you followed Lesson 4's setup, those packages are already available. Otherwise, from the repository root:

```bash
uv sync
uv pip install --python .venv/bin/python torch torchvision matplotlib
```

PyTorch's accelerator build depends on your system. Use the matching installation option from the [official PyTorch installer](https://pytorch.org/get-started/locally/) if you want CUDA support.

## Run

From the repository root:

```bash
uv run python examples/02-modern-cnns/lesson-07/train_cifar10.py
```

The first run downloads CIFAR-10 under `data/cifar10/`. The script uses 45,000 images for training, 5,000 for validation, and the official 10,000-image test set for final evaluation. It trains for five epochs, using CUDA when available and CPU otherwise.

## Expected outputs

The script prints the shape after each CNN layer and training/validation metrics per epoch, followed by final test metrics. Random crops and horizontal flips are applied to training images; validation and test images stay unchanged. It saves `sample_images.png`, `augmentation_examples.png`, `rgb_filters.png`, and `cifar10_cnn.pth` under `examples/02-modern-cnns/lesson-07/outputs/`.

## Limitations

This is a small teaching baseline with no data augmentation or Batch Normalization. Runtime and accuracy vary with hardware and initialization. The example does not tune the architecture or guarantee a particular accuracy.
