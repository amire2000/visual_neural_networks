# Lesson 0: MNIST examples

## Prerequisites

From the repository root, prepare the course environment and install PyTorch, TorchVision, and Matplotlib. Choose the PyTorch build that matches your CPU/GPU using the [official install selector](https://pytorch.org/get-started/locally/).

```bash
uv sync
uv pip install --python .venv/bin/python torch torchvision matplotlib
```

## Download both MNIST splits

```bash
uv run python examples/00-setup/download_mnist.py
```

The script downloads the training and test data under `data/mnist/` and prints their sizes, the shape of one image, and its label.

## Load one image and label

```bash
uv run python examples/00-setup/load_mnist_sample.py
```

The script gets the first training example with `train_data[0]`, prints its digit label and image tensor shape, and displays the grayscale image. If the dataset has not been downloaded yet, TorchVision downloads the training split first.
