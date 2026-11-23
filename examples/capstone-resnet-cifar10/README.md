# CIFAR-10 ResNet paper-to-code capstone

## Prerequisites

Use the repository `.venv` with PyTorch, TorchVision, and Matplotlib installed. Follow [Setup Lab](../../docs/lesson-00-setup-lab/index.md) if needed. The first run downloads CIFAR-10 into `data/cifar10/`. CUDA is used when available; CPU-only systems also run the project.

## Run

From the repository root, first check model shapes, then train the ResNet-20:

```bash
uv run python examples/capstone-resnet-cifar10/resnet_cifar.py
uv run python examples/capstone-resnet-cifar10/train.py --model resnet20
```

Train the plain CNN comparison with the same split and settings:

```bash
uv run python examples/capstone-resnet-cifar10/train.py --model baseline
```

## Expected outputs

The shape check prints `(2, 3, 32, 32) → (2, 10)`. Training prints epoch loss and accuracy, then the best validation accuracy and final test accuracy. It saves `cifar10_samples.png`, `training_curves.png`, and a model-specific best checkpoint under `examples/capstone-resnet-cifar10/outputs/`.

## Limitations

The five-epoch run is a short learning experiment, not a reproduction of the paper's training recipe or reported accuracy. The training augmentation and optimizer are shared by both models for a fair architecture comparison, but this does not make the models equal in size or compute. Results depend on hardware and runtime.
