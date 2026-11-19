"""Load one MNIST training image and its digit label."""

from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"


def main():
    """Download MNIST if needed, print one sample, and display its image."""
    train_data = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    image, label = train_data[0]
    print("Label:", label)
    print("Image shape:", tuple(image.shape))
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(f"MNIST label: {label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
