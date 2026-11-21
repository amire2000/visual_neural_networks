"""Run a trained MNIST Vision Transformer on a random test batch and save a figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_vit import MNISTVisionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_vit.pth"
BATCH_SIZE = 16
RANDOM_SEED = 77
GRID_COLUMNS = 4


def save_predictions(model, loader, device):
    """Predict one shuffled test batch and save images labeled with predictions."""
    images, labels = next(iter(loader))
    model.eval()
    with torch.no_grad():
        predictions = model(images.to(device)).argmax(dim=1).cpu()
    accuracy = 100.0 * (predictions == labels).float().mean().item()

    figure, axes = plt.subplots(BATCH_SIZE // GRID_COLUMNS, GRID_COLUMNS, figsize=(9, 9))
    for index, axis in enumerate(axes.flat):
        correct = predictions[index].item() == labels[index].item()
        axis.imshow(images[index, 0], cmap="gray")
        axis.set_title(
            f"True {labels[index].item()} | Pred {predictions[index].item()}",
            color="green" if correct else "crimson",
        )
        axis.axis("off")
    figure.suptitle(f"MNIST ViT test batch — {accuracy:.1f}% correct")
    figure.tight_layout()
    result_path = OUTPUT_DIR / "vit_predictions.png"
    figure.savefig(result_path, dpi=160)
    print(f"Batch accuracy: {accuracy:.2f}% ({int((predictions == labels).sum())}/{len(labels)})")
    print(f"Saved prediction grid: {result_path}")
    plt.show()


def main():
    """Load the best checkpoint and visualize predictions on unseen test images."""
    assert CHECKPOINT_PATH.exists(), "Run train_vit_mnist.py before inference."
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    model = MNISTVisionTransformer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    save_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()
