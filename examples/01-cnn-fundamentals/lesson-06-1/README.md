# Lesson 6.1: TensorBoard for MNIST training

## Prerequisites

Python 3.11 or later, the course `.venv`, and PyTorch, TorchVision, and TensorBoard. Matplotlib is optional for this example; TensorBoard renders its own dashboard.

From the repository root, install the packages into the course environment (choose the CPU or GPU PyTorch build appropriate for your system):

```bash
uv sync
uv pip install --python .venv/bin/python torch torchvision tensorboard
```

## Run a training experiment

From the repository root:

```bash
uv run python examples/01-cnn-fundamentals/lesson-06-1/visualize_training.py --learning-rate 0.001 --run-name mnist_lr_0p001
```

The first run downloads MNIST under `data/mnist/`. Each run trains for 20 epochs and saves its TensorBoard event files and best validation checkpoint under `examples/01-cnn-fundamentals/lesson-06-1/outputs/tensorboard/`.

## Open TensorBoard

In another terminal, from the repository root:

```bash
uv run tensorboard --logdir examples/01-cnn-fundamentals/lesson-06-1/outputs/tensorboard/runs --host 127.0.0.1
```

Open `http://localhost:6006` in a browser on the same machine. Keep the dashboard bound to localhost unless you have configured a secure remote-access method.

## Compare learning rates

Run independent configurations with separate run names:

```bash
uv run python examples/01-cnn-fundamentals/lesson-06-1/visualize_training.py --learning-rate 0.0001 --run-name mnist_lr_0p0001
uv run python examples/01-cnn-fundamentals/lesson-06-1/visualize_training.py --learning-rate 0.01 --run-name mnist_lr_0p01
```

The seed, architecture, data split, batch size, and epoch count stay fixed. Only the learning rate and run name change.

## Expected outputs

- TensorBoard event data under `outputs/tensorboard/runs/<run-name>/`
- A run-settings summary in TensorBoard's **Text** tab, including the learning rate and fixed experiment settings
- The best-validation-loss checkpoint under `outputs/tensorboard/checkpoints/<run-name>.pth`
- Console output with train and validation metrics for each epoch. The separate `evaluate_best.py` command prints final test loss and accuracy after a run is selected.

## Limitations

The example logs one scalar learning rate per epoch and histograms every five epochs. It records convolution weights in those histograms, not biases or classifier weights. Training time and accuracy vary by hardware. Each run name maps to one event directory and checkpoint; rerunning the same name reuses that directory. Compare/select using validation metrics and run `evaluate_best.py` only for the selected checkpoint.
