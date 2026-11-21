"""Train a small Vision Transformer and save its best validation checkpoint."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mnist_vit import IMAGE_SIZE, INPUT_CHANNELS, NUM_CLASSES, MNISTVisionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BATCH_SIZE = 128
EPOCHS = 5
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
LEARNING_RATE = 0.001
RANDOM_SEED = 21
NUM_WORKERS = 0


def make_loaders():
    """Download MNIST and return deterministic training and validation loaders."""
    dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    train_set, validation_set = random_split(
        dataset,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, validation_loader


def evaluate(model, loader, loss_function, device):
    """Return average loss and accuracy without changing model weights."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def check_model_output(model, device):
    """Assert that one grayscale image produces ten class scores."""
    sample_image = torch.zeros(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        sample_logits = model(sample_image)
    assert tuple(sample_logits.shape) == (1, NUM_CLASSES)


def save_curves(history):
    """Save training and validation loss and accuracy as a PNG graph."""
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Percent correct", ylim=(0, 100))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(figure)


def main():
    """Train the model, save the best validation weights, and plot learning curves."""
    torch.manual_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders()
    model = MNISTVisionTransformer().to(device)
    check_model_output(model, device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_validation_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Predict, measure error, backpropagate, then update weights.
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss_total / train_total,
            "train_accuracy": 100.0 * train_correct / train_total,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{EPOCHS} | train loss {row['train_loss']:.4f} "
            f"| train acc {row['train_accuracy']:.2f}% | val acc {validation_accuracy:.2f}%"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), OUTPUT_DIR / "best_vit.pth")

    save_curves(history)
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
    print(f"Saved checkpoint and graph in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
