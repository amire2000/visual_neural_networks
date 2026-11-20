"""Adapt a pretrained ResNet-18 head and compare fine-tuning choices."""

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
INPUT_SIZE = 224
SEED = 10


# --8<-- [start:model_setup]
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 10
INPUT_CHANNELS = 3
BATCH_SIZE = 1


def make_transfer_model():
    """Load pretrained visual features and attach a CIFAR-10 output head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
# --8<-- [end:model_setup]


def count_trainable_parameters(model):
    """Count parameters that an optimizer will update."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_comparison(model, logits):
    """Save a visual comparison of frozen and partially fine-tuned stages."""
    head_only = count_trainable_parameters(model)
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    partial = count_trainable_parameters(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    values = (head_only, partial)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(("New head only", "Layer 4 + head"), values, color=("#238b70", "#e07a28"))
    axes[0].set_title("Parameters updated during training")
    axes[0].set_ylabel("Trainable parameters")
    axes[0].ticklabel_format(axis="y", style="plain")
    for index, value in enumerate(values):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")
    axes[1].axis("off")
    axes[1].set_title("Adapted model output")
    axes[1].text(0.5, 0.65, f"Input: 1 × 3 × {INPUT_SIZE} × {INPUT_SIZE}", ha="center", transform=axes[1].transAxes)
    axes[1].text(0.5, 0.45, f"Logits: {list(logits.shape)}", ha="center", transform=axes[1].transAxes)
    axes[1].text(0.5, 0.25, f"Total parameters: {total:,}", ha="center", transform=axes[1].transAxes)
    figure.suptitle("Transfer learning: reuse features, choose what to update")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "transfer_learning_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    """Run a shape check and save trainable-parameter comparisons."""
    torch.manual_seed(SEED)
    model = make_transfer_model().eval()
    image_batch = torch.rand(BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        logits = model(image_batch)
    print("Output logits:", list(logits.shape))
    print("Head-only trainable parameters:", f"{count_trainable_parameters(model):,}")
    assert list(logits.shape) == [BATCH_SIZE, NUM_CLASSES]
    save_comparison(model, logits)
    print("Saved:", OUTPUT_DIR / "transfer_learning_comparison.png")


if __name__ == "__main__":
    main()
