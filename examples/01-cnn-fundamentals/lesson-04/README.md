# Lesson 4: Train a CNN on MNIST

## Prerequisites

Python 3.11 or later, `uv`, and the course `.venv`. This example uses PyTorch, TorchVision, and Matplotlib. PyTorch's installation command varies by operating system and CPU/GPU choice; use the [official installer](https://pytorch.org/get-started/locally/) to select the appropriate build.

From the repository root, prepare the environment:

```bash
uv sync
uv pip install --python .venv/bin/python torch torchvision matplotlib
```

For an accelerator-specific PyTorch build, use the matching index option from the official installer with the `uv pip install --python .venv/bin/python` command.

## Run

From the repository root:

```bash
uv run python examples/01-cnn-fundamentals/lesson-04/train_mnist.py
```

The first run downloads MNIST under `data/mnist/`. By default, 54,000 examples train the model, 6,000 are used for validation, and the 10,000 test images are held out until the end. Training runs for five epochs and uses CUDA when available, otherwise CPU.

## Expected outputs

The script prints tensor shapes, train and validation metrics each epoch, and final test metrics. It saves:

- `mnist_cnn.pth`
- `mnist_sample.png`
- `feature_maps.png`
- `kernels_before_after.png`
- `training_history.csv`
- `training_curves.png`

These files are written to `examples/01-cnn-fundamentals/lesson-04/outputs/`.

## Limitations

Training time and accuracy depend on hardware and the installed PyTorch build. The example uses a fixed architecture and five epochs by default; it does not tune hyperparameters or guarantee a particular accuracy.
