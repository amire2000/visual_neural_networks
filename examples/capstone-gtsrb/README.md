# GTSRB capstone example

## Prerequisites

Use the repository `.venv` with PyTorch, TorchVision, and Matplotlib installed. Follow [Setup Lab](../../docs/lesson-00-setup-lab/index.md) if needed. The optional ResNet-18 run also downloads pretrained weights.

## Run

From the repository root:

```bash
uv run python examples/capstone-gtsrb/train_gtsrb.py --help
uv run python examples/capstone-gtsrb/train_gtsrb.py
```

The first run downloads GTSRB into `data/gtsrb/`. To use pretrained ResNet-18 instead of the default custom CNN:

```bash
uv run python examples/capstone-gtsrb/train_gtsrb.py --model resnet18
```

## Expected outputs

The script prints the device, class split sizes, tensor shapes, epoch metrics, and final official test loss and accuracy. It saves `class_samples.png`, `training_curves.png`, `confusion_matrix.png`, `test_errors.png`, and the best validation checkpoint in `examples/capstone-gtsrb/outputs/`.

## Limitations

This is an educational image-classification project. The validation split groups files using their recording-track filename prefix; confirm the split protocol against the dataset version you download. The model receives sign crops and does not locate signs in full road scenes. A short run is not a safety evaluation, and accuracy can vary by hardware, random seed, and training duration. Do not use this model to make real driving decisions.
