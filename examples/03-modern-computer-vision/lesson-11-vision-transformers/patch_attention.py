"""Visualize image patches and a tiny, color-based self-attention example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
IMAGE_SIZE = 128
PATCH_SIZE = 16
GRID_SIDE = IMAGE_SIZE // PATCH_SIZE
QUERY_ROW = 4
QUERY_COLUMN = 2
ATTENTION_SCALE = 8.0


# --8<-- [start:attention_step]
def attend_to_query(patch_tokens, query_index):
    """Return attention weights and the weighted value for one patch token."""
    normalized_tokens = functional.normalize(patch_tokens, dim=1)
    query = normalized_tokens[query_index]
    scores = normalized_tokens @ query * ATTENTION_SCALE
    weights = torch.softmax(scores, dim=0)
    mixed_value = weights @ patch_tokens
    return weights, mixed_value
# --8<-- [end:attention_step]


def make_toy_image():
    """Draw a simple colorful scene whose objects align with image patches."""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (204, 229, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 80, IMAGE_SIZE, IMAGE_SIZE), fill=(90, 170, 100))
    draw.ellipse((88, 8, 120, 40), fill=(255, 210, 55))
    draw.rectangle((32, 48, 80, 104), fill=(205, 65, 55))
    draw.polygon(((24, 50), (56, 24), (88, 50)), fill=(110, 45, 45))
    draw.rectangle((48, 64, 64, 80), fill=(70, 130, 210))
    return image


# --8<-- [start:patchify]
def patchify(image):
    """Split an RGB image into non-overlapping patches and mean-color tokens."""
    pixels = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
    patches = torch.stack(
        [
            pixels[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            for y in range(0, IMAGE_SIZE, PATCH_SIZE)
            for x in range(0, IMAGE_SIZE, PATCH_SIZE)
        ]
    )
    return patches, patches.mean(dim=(1, 2))
# --8<-- [end:patchify]


def save_attention_figure(image, patch_tokens, weights, mixed_value):
    """Save a four-panel view of the image, tokens, attention, and output."""
    query_index = QUERY_ROW * GRID_SIDE + QUERY_COLUMN
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].add_patch(
        plt.Rectangle(
            (QUERY_COLUMN * PATCH_SIZE, QUERY_ROW * PATCH_SIZE),
            PATCH_SIZE,
            PATCH_SIZE,
            fill=False,
            edgecolor="yellow",
            linewidth=3,
        )
    )
    axes[0].text(
        QUERY_COLUMN * PATCH_SIZE + 2,
        QUERY_ROW * PATCH_SIZE - 3,
        "query",
        color="black",
        backgroundcolor="yellow",
        fontsize=8,
    )

    axes[1].imshow(image)
    for row in range(GRID_SIDE):
        for column in range(GRID_SIDE):
            index = row * GRID_SIDE + column
            axes[1].text(
                column * PATCH_SIZE + PATCH_SIZE / 2,
                row * PATCH_SIZE + PATCH_SIZE / 2,
                str(index),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
    axes[1].set_title("Split into 64 tokens")

    attention = weights.reshape(GRID_SIDE, GRID_SIDE).numpy()
    heatmap = axes[2].imshow(attention, cmap="magma")
    axes[2].set_title(f"Attention from token {query_index}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    figure.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    color = mixed_value.clamp(0, 1).numpy()
    axes[3].imshow(np.tile(color, (PATCH_SIZE * 4, PATCH_SIZE * 4, 1)))
    axes[3].set_title("Attention-weighted value")
    axes[3].text(
        0.5,
        0.5,
        "Weighted RGB\n" + ", ".join(f"{value:.2f}" for value in color),
        transform=axes[3].transAxes,
        ha="center",
        va="center",
        color="black",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    for axis in (axes[0], axes[1], axes[3]):
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "patch_attention.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main():
    """Create patch tokens, attend from one patch, and display the visual result."""
    image = make_toy_image()
    patches, patch_tokens = patchify(image)
    query_index = QUERY_ROW * GRID_SIDE + QUERY_COLUMN
    weights, mixed_value = attend_to_query(patch_tokens, query_index)
    print("Image:", (3, IMAGE_SIZE, IMAGE_SIZE))
    print("Patch tensor:", tuple(patches.shape))
    print("Toy token tensor:", tuple(patch_tokens.shape))
    print("Attention weights:", tuple(weights.shape), "sum:", weights.sum().item())
    save_attention_figure(image, patch_tokens, weights, mixed_value)
    print("Saved and displayed:", OUTPUT_DIR / "patch_attention.png")


if __name__ == "__main__":
    main()
