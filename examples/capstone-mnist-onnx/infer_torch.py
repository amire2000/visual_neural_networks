"""Run the saved MNIST MLP with PyTorch, optionally using TorchAO int8 quantization."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_mnist_mlp import MNISTMLP, OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
INFERENCE_BATCH_SIZE = 16
GRID_COLUMNS = 4
GRID_ROWS = 4
SEED = 31


def parse_args():
    """Parse whether to quantize the MLP with TorchAO before inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use TorchAO dynamic int8 quantization on the model's Linear weights.",
    )
    return parser.parse_args()


def load_model(quantized):
    """Load the trained MLP and optionally apply TorchAO dynamic int8 quantization."""
    checkpoint_path = OUTPUT_DIR / "best_mlp.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run train_mnist_mlp.py first."
        )
    model = MNISTMLP().cpu().eval()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    if quantized:
        # --8<-- [start:quantization]
        try:
            from torchao.quantization import (
                Int8DynamicActivationInt8WeightConfig,
                quantize_,
            )
        except ImportError as error:
            raise SystemExit(
                "Quantized inference needs TorchAO. Install a TorchAO build "
                "compatible with this PyTorch installation."
            ) from error
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        # --8<-- [end:quantization]
    return model.eval()


def make_test_batch():
    """Load a reproducible shuffled batch from MNIST's official test split."""
    dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    return next(iter(loader))


def save_prediction_grid(images, labels, logits, quantized):
    """Save and display the true and predicted labels for one test batch."""
    predictions = logits.argmax(dim=1)
    confidences = logits.softmax(dim=1).amax(dim=1)
    batch_accuracy = 100.0 * (predictions == labels).float().mean().item()
    figure, axes = plt.subplots(
        GRID_ROWS, GRID_COLUMNS, figsize=(9, 9), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        is_correct = predictions[index] == labels[index]
        axis.imshow(images[index, 0], cmap="gray")
        axis.set_title(
            f"true {labels[index].item()} · pred {predictions[index].item()} "
            f"({confidences[index].item():.0%})",
            color="#15803d" if is_correct else "#dc2626",
            fontsize=9,
        )
        axis.axis("off")
    mode_name = "TorchAO dynamic int8" if quantized else "PyTorch FP32"
    figure.suptitle(f"{mode_name} · {batch_accuracy:.1f}% batch accuracy")
    figure.tight_layout()
    filename = "torch_quantized_predictions.png" if quantized else "torch_predictions.png"
    result_path = OUTPUT_DIR / filename
    figure.savefig(result_path, dpi=160, bbox_inches="tight")
    print("Mode:", mode_name)
    print("Batch accuracy:", f"{batch_accuracy:.2f}%")
    print("Saved prediction grid:", result_path)
    plt.show()


def main():
    """Run a test batch through PyTorch and save the prediction grid."""
    arguments = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = make_test_batch()
    # --8<-- [start:torch_inference]
    model = load_model(arguments.quantized)
    with torch.inference_mode():
        logits = model(images)
    # --8<-- [end:torch_inference]
    print("Input shape:", list(images.shape), "| Output shape:", list(logits.shape))
    save_prediction_grid(images, labels, logits, arguments.quantized)


if __name__ == "__main__":
    main()
