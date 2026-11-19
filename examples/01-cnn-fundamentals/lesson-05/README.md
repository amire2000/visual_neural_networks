# Lesson 5: Visualize learned CNN features

## Prerequisites

Python 3.11 or later, the course `.venv`, and the PyTorch, TorchVision, and Matplotlib packages from Lesson 4. Train the CNN first so its checkpoint exists at `examples/01-cnn-fundamentals/lesson-04/outputs/mnist_cnn.pth`.

## Run

From the repository root:

```bash
uv run python examples/01-cnn-fundamentals/lesson-05/visualize_cnn.py
```

The script reuses the MNIST test set under `data/mnist/` and saves its plots under `examples/01-cnn-fundamentals/lesson-05/outputs/`.

## Expected outputs

- `first_layer_maps.png`
- `second_layer_maps.png`
- `trained_kernels.png`

## Limitations

Feature maps are intermediate activations, not digit labels or probabilities. The script displays the first eight maps from each convolutional layer and all 16 first-layer kernels.
