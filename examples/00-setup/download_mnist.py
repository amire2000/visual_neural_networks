"""Download both MNIST splits and print a quick data sanity check."""

from pathlib import Path

from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"


def main():
    """Download the training and test splits and report their basic shapes."""
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    image, label = train_data[0]

    print(f"Training images: {len(train_data)}")
    print(f"Test images: {len(test_data)}")
    print(f"One image tensor: {tuple(image.shape)}")
    print(f"First image label: {label}")
    print(f"MNIST files: {DATA_DIR}")


if __name__ == "__main__":
    main()
