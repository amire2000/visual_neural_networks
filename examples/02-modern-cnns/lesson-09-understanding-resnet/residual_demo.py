"""Show the identity path, learned change, and sum in a residual block."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEMO_CHANNELS = 16
FEATURE_MAP_SIZE = 8


# --8<-- [start:residual_block]
class ResidualBlock(torch.nn.Module):
    """Add a learned two-convolution change to same-shaped input features."""

    def __init__(self, channels):
        """Create a residual block that preserves its channel and image size."""
        super().__init__()
        self.change = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels),
        )
        self.output_activation = torch.nn.ReLU()

    def forward(self, features):
        """Return the input features plus the learned residual change."""
        return self.output_activation(features + self.change(features))
# --8<-- [end:residual_block]


def save_residual_visual(shortcut, residual):
    """Plot the two paths and their element-wise sum using a shared color scale."""
    combined = shortcut + residual
    values = torch.cat((shortcut.flatten(), residual.flatten(), combined.flatten()))
    minimum, maximum = values.min().item(), values.max().item()
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for axis, matrix, title in zip(axes, (shortcut, residual, combined), ("Input x", "Learned change F(x)", "Output x + F(x)")):
        axis.imshow(matrix, cmap="coolwarm", vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column].item():.1f}", ha="center", va="center", fontsize=8)
    figure.suptitle("Residual addition keeps the input and adds a learned change")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "residual_addition.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return combined


def main():
    """Create a tiny feature-map example and save the residual addition."""
    block = ResidualBlock(channels=DEMO_CHANNELS).eval()
    sample_features = torch.rand(1, DEMO_CHANNELS, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
    with torch.inference_mode():
        block_output = block(sample_features)
    assert block_output.shape == sample_features.shape

    shortcut = torch.tensor(
        [[0.0, 0.2, 0.4, 0.1], [0.1, 0.8, 0.7, 0.2], [0.0, 0.6, 0.9, 0.3], [0.0, 0.1, 0.3, 0.1]]
    )
    residual = torch.tensor(
        [[0.0, 0.1, -0.1, 0.0], [0.1, 0.2, 0.0, -0.1], [0.0, -0.1, 0.2, 0.0], [0.0, 0.0, 0.1, 0.0]]
    )
    output = save_residual_visual(shortcut, residual)
    assert torch.equal(output, shortcut + residual)
    print("Input, change, and output shapes:", list(shortcut.shape), list(residual.shape), list(output.shape))
    print("Saved:", OUTPUT_DIR / "residual_addition.png")


if __name__ == "__main__":
    main()
