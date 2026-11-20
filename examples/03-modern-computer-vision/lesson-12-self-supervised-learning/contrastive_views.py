"""Learn to match two augmented views of the same unlabeled image."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
IMAGE_SIDE = 32
IMAGE_COUNT = 72
BATCH_SIZE = 24
VISUALIZED_COUNT = 12
HIDDEN_UNITS = 64
EMBEDDING_SIZE = 16
EPOCHS = 20
TEMPERATURE = 0.2
LEARNING_RATE = 0.001
MAX_VIEW_SHIFT = 1
VIEW_NOISE = 0.02
SEED = 11


class TinyEncoder(nn.Module):
    """Map a small grayscale image to a normalized visual embedding."""

    def __init__(self):
        """Create a compact fully connected image encoder."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIDE * IMAGE_SIDE, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, EMBEDDING_SIZE),
        )

    def forward(self, images):
        """Return one unit-length feature vector for each image."""
        return functional.normalize(self.layers(images), dim=1)


def make_unlabeled_images():
    """Create shifted line and loop images without returning their types."""
    images = []
    for index in range(IMAGE_COUNT):
        image = Image.new("L", (IMAGE_SIDE, IMAGE_SIDE), 0)
        draw = ImageDraw.Draw(image)
        offset = index % 5 - 2
        pattern = index % 3
        if pattern == 0:
            draw.line((5, 15 + offset, 27, 15 + offset), fill=255, width=4)
        elif pattern == 1:
            draw.line((15 + offset, 5, 15 + offset, 27), fill=255, width=4)
        else:
            draw.ellipse((7 + offset, 7, 25 + offset, 25), outline=255, width=4)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).float()
        images.append(pixels.reshape(1, IMAGE_SIDE, IMAGE_SIDE) / 255.0)
    return torch.stack(images)


def augment(images):
    """Make a changed view of every image while preserving its main pattern."""
    views = []
    for image in images:
        view = image.clone()
        if torch.rand(()) > 0.5:
            view = view.flip(-1)
        shift_y = int(torch.randint(-MAX_VIEW_SHIFT, MAX_VIEW_SHIFT + 1, ()))
        shift_x = int(torch.randint(-MAX_VIEW_SHIFT, MAX_VIEW_SHIFT + 1, ()))
        view = torch.roll(view, shifts=(shift_y, shift_x), dims=(-2, -1))
        view = (view + VIEW_NOISE * torch.randn_like(view)).clamp(0, 1)
        views.append(view)
    return torch.stack(views)


# --8<-- [start:contrastive_loss]
def contrastive_loss(encoder, first_view, second_view):
    """Reward matching views of one image while separating other images."""
    batch_size = len(first_view)
    embeddings = encoder(torch.cat((first_view, second_view)))
    similarities = embeddings @ embeddings.T / TEMPERATURE
    similarities.fill_diagonal_(float("-inf"))
    positive_indices = (torch.arange(2 * batch_size) + batch_size) % (2 * batch_size)
    return functional.cross_entropy(similarities, positive_indices)
# --8<-- [end:contrastive_loss]


def make_similarity_matrix(encoder, first_view, second_view):
    """Return pairwise cosine similarities for two batches of image views."""
    with torch.no_grad():
        embeddings = encoder(torch.cat((first_view, second_view)))
        paired_order = torch.arange(2 * len(first_view)).reshape(2, -1).T.flatten()
        embeddings = embeddings[paired_order]
        return embeddings @ embeddings.T


def save_visualization(images, view_a, view_b, before, after, losses):
    """Save and display the views, training curve, and similarity matrices."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    montage = torch.cat((images[0, 0], view_a[0, 0], view_b[0, 0]), dim=1)
    axes[0, 0].imshow(montage, cmap="gray")
    axes[0, 0].set_title("One image · view A · view B")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[0, 1].plot(range(1, len(losses) + 1), losses, color="#15803d")
    axes[0, 1].set_title("Contrastive training loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss (lower means easier view matching)")

    for axis, matrix, title in (
        (axes[1, 0], before, "Before training"),
        (axes[1, 1], after, "After training"),
    ):
        heatmap = axis.imshow(matrix, cmap="viridis", vmin=-1, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("Image views · paired A/B order")
        axis.set_ylabel("Image views · paired A/B order")
        figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Self-supervision: match views from the same source image")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "contrastive_views.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main():
    """Train the toy image encoder without labels and display the result."""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    images = make_unlabeled_images()
    visual_images = images[:VISUALIZED_COUNT]
    visual_view_a = augment(visual_images)
    visual_view_b = augment(visual_images)
    encoder = TinyEncoder()
    before = make_similarity_matrix(encoder, visual_view_a, visual_view_b)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    losses = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(images))
        epoch_losses = []
        for indices in order.split(BATCH_SIZE):
            first_view = augment(images[indices])
            second_view = augment(images[indices])
            loss = contrastive_loss(encoder, first_view, second_view)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Epoch {epoch + 1}/{EPOCHS}: contrastive loss={losses[-1]:.3f}")

    after = make_similarity_matrix(encoder, visual_view_a, visual_view_b)
    print("Unlabeled image batch:", tuple(images.shape))
    print("Embedding batch:", (2 * VISUALIZED_COUNT, EMBEDDING_SIZE))
    save_visualization(
        visual_images, visual_view_a, visual_view_b, before, after, losses
    )
    print("Saved and displayed:", OUTPUT_DIR / "contrastive_views.png")


if __name__ == "__main__":
    main()
