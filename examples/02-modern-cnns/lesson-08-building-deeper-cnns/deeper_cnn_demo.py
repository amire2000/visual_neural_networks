"""Compare a CIFAR-10 CNN classifier head with global average pooling."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
INPUT_CHANNELS = 3
IMAGE_SIZE = 32
NUM_CLASSES = 10
CHANNELS = (32, 64, 128)
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2
SEED = 8


# --8<-- [start:model]
class DeeperCNN(nn.Module):
    """Build three convolution blocks with normalization and global pooling."""

    def __init__(self):
        """Create feature blocks and a small ten-class output head."""
        super().__init__()
        blocks = []
        in_channels = INPUT_CHANNELS
        for out_channels in CHANNELS:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, KERNEL_SIZE, padding=PADDING),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(POOL_SIZE),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(CHANNELS[-1], NUM_CLASSES)

    def forward(self, images):
        """Return ten class scores for each input image."""
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)
# --8<-- [end:model]


def save_comparison():
    """Save the tensor-shape path and classifier parameter comparison."""
    shape_labels = ["32 × 32 × 3", "16 × 16 × 32", "8 × 8 × 64", "4 × 4 × 128", "128 features"]
    head_parameters = {
        "Flatten + Linear\n(2,048 → 10)": (CHANNELS[-1] * 4 * 4 + 1) * NUM_CLASSES,
        "Global average pool\n(128 → 10)": (CHANNELS[-1] + 1) * NUM_CLASSES,
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].axis("off")
    axes[0].set_title("A park-camera frame through a deeper CNN")
    axes[0].text(0.5, 0.92, "32 × 32 RGB input", ha="center", weight="bold", transform=axes[0].transAxes)
    for index, label in enumerate(shape_labels[1:]):
        y = 0.77 - index * 0.17
        axes[0].text(
            0.5, y, f"↓  {label}", ha="center", va="center", fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#e8f1ff", "edgecolor": "#4c78a8"},
            transform=axes[0].transAxes,
        )

    axes[1].bar(head_parameters.keys(), head_parameters.values(), color=("#e07a28", "#238b70"))
    axes[1].set_title("Classifier weights for 10 classes")
    axes[1].set_ylabel("Parameters, including bias")
    for index, value in enumerate(head_parameters.values()):
        axes[1].text(index, value, f"{value:,}", ha="center", va="bottom")
    figure.suptitle("Deeper features, compact classifier")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "deeper_cnn_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    """Print shape changes and save a visual classifier-size comparison."""
    torch.manual_seed(SEED)
    model = DeeperCNN().eval()
    image = torch.rand(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    print("Input:", list(image.shape))
    with torch.inference_mode():
        features = image
        for layer in model.features:
            features = layer(features)
            if isinstance(layer, nn.MaxPool2d):
                print("After pooling:", list(features.shape))
        logits = model(image)
    print("Output logits:", list(logits.shape))
    print("Classifier parameters after global average pooling:", sum(p.numel() for p in model.classifier.parameters()))
    assert list(logits.shape) == [1, NUM_CLASSES]
    save_comparison()
    print("Saved:", OUTPUT_DIR / "deeper_cnn_comparison.png")


if __name__ == "__main__":
    main()
