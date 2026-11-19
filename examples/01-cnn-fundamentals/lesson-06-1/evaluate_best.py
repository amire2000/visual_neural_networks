"""Evaluate one selected MNIST checkpoint on the official test split."""

import argparse
import re

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from visualize_training import DATA_DIR, OUTPUT_DIR, SimpleCNN, evaluate


def main():
    """Evaluate the selected checkpoint once on MNIST's held-out test data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name may contain only letters, numbers, dots, underscores, and hyphens")

    checkpoint = OUTPUT_DIR / "checkpoints" / f"{args.run_name}.pth"
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")

    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    loss, accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
