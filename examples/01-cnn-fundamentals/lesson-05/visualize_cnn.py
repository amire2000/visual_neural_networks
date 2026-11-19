"""Visualize learned MNIST activations from the Lesson 4 CNN checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
CHECKPOINT = PROJECT_ROOT / "examples/01-cnn-fundamentals/lesson-04/outputs/mnist_cnn.pth"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# --8<-- [start:model]
class SimpleCNN(nn.Module):
    """Match the Lesson 4 network so its saved weights load correctly."""

    def __init__(self):
        """Create the same layers and dimensions used during Lesson 4 training."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """Return ten raw digit scores for each image in the batch."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(self.flatten(x))
# --8<-- [end:model]


# --8<-- [start:load]
def load_model():
    """Load the trained Lesson 4 checkpoint on the CPU."""
    model = SimpleCNN()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()
    return model
# --8<-- [end:load]


# --8<-- [start:examples]
def select_examples(dataset, wanted_digits=(1, 5, 8)):
    """Return the first test image found for each requested digit."""
    selected = {}
    for image, label in dataset:
        if label in wanted_digits and label not in selected:
            selected[label] = image
            if len(selected) == len(wanted_digits):
                break
    return selected
# --8<-- [end:examples]


# --8<-- [start:activations]
def get_feature_maps(model, images):
    """Return first- and second-layer activations without tracking gradients."""
    with torch.no_grad():
        first_maps = model.relu(model.conv1(images))
        second_maps = model.relu(model.conv2(model.pool(first_maps)))
    return first_maps, second_maps
# --8<-- [end:activations]


# --8<-- [start:plot_maps]
def save_map_grid(images, labels, maps, filename, layer_name, map_count=8):
    """Save input images beside a selection of their feature maps."""
    fig, axes = plt.subplots(len(labels), map_count + 1, figsize=(14, 5))
    for row, label in enumerate(labels):
        axes[row, 0].imshow(images[row, 0], cmap="gray")
        axes[row, 0].set_title(f"Input: {label}")
        axes[row, 0].axis("off")
        for channel in range(map_count):
            axes[row, channel + 1].imshow(maps[row, channel], cmap="gray")
            axes[row, channel + 1].set_title(f"Map {channel + 1}")
            axes[row, channel + 1].axis("off")
    fig.suptitle(layer_name)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
# --8<-- [end:plot_maps]


def save_kernel_grid(model):
    """Save the trained first-layer 3 by 3 kernels as a 4 by 4 grid."""
    weights = model.conv1.weight.detach().cpu()
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for index, axis in enumerate(axes.flat):
        axis.imshow(weights[index, 0], cmap="coolwarm")
        axis.set_title(f"Filter {index + 1}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "trained_kernels.png")
    plt.close(fig)


def main():
    """Load the checkpoint and save feature-map and kernel plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    selected = select_examples(dataset)
    labels = (1, 5, 8)
    images = torch.stack([selected[label] for label in labels])

    first_maps, second_maps = get_feature_maps(model, images)
    assert tuple(images.shape) == (3, 1, 28, 28)
    assert tuple(first_maps.shape) == (3, 16, 28, 28)
    assert tuple(second_maps.shape) == (3, 32, 14, 14)

    save_map_grid(images, labels, first_maps, OUTPUT_DIR / "first_layer_maps.png", "Conv1 feature maps")
    save_map_grid(images, labels, second_maps, OUTPUT_DIR / "second_layer_maps.png", "Conv2 feature maps")
    save_kernel_grid(model)
    print("Saved first- and second-layer maps and trained kernels to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
