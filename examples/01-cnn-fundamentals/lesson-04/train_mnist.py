"""Train a small PyTorch CNN on MNIST and save visual diagnostics."""

from pathlib import Path

import csv

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BATCH_SIZE = 64
EPOCHS = 5
TRAIN_SIZE = 54000
VALIDATION_SIZE = 6000


# --8<-- [start:model]
class SimpleCNN(nn.Module):
    """Classify 28 by 28 grayscale images into ten digit classes."""

    def __init__(self):
        """Create the convolutional and classification layers."""
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


# --8<-- [start:loaders]
def make_loaders():
    """Create separate training, validation, and test loaders for MNIST."""
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_subset, validation_subset, _ = random_split(
        train_data,
        [TRAIN_SIZE, VALIDATION_SIZE, len(train_data) - TRAIN_SIZE - VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return test_data, train_loader, validation_loader, test_loader
# --8<-- [end:loaders]


# --8<-- [start:inspect]
def inspect_shapes(model, device):
    """Print each layer's output shape for one dummy image."""
    model.eval()
    x = torch.zeros(1, 1, 28, 28, device=device)
    print("Input:", list(x.shape))
    with torch.no_grad():
        x = model.conv1(x)
        print("After Conv1:", list(x.shape))
        x = model.relu(x)
        print("After ReLU:", list(x.shape))
        x = model.pool(x)
        print("After Pool1:", list(x.shape))
        x = model.conv2(x)
        print("After Conv2:", list(x.shape))
        x = model.relu(x)
        x = model.pool(x)
        print("After Pool2:", list(x.shape))
        x = model.flatten(x)
        print("After Flatten:", list(x.shape))
        x = model.fc(x)
        print("Output:", list(x.shape))
# --8<-- [end:inspect]


# --8<-- [start:train]
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Update the model using every batch in the training loader once."""
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
# --8<-- [end:train]


# --8<-- [start:evaluate]
def evaluate(model, loader, loss_fn, device):
    """Return mean loss and accuracy without updating the model."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total
# --8<-- [end:evaluate]


def save_training_history(history):
    """Write per-epoch training and validation metrics to a CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"]
    with (OUTPUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(history):
    """Plot training and validation loss and accuracy by epoch."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Percent correct", ylim=(0, 100))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png")
    plt.close(fig)


# --8<-- [start:visualize]
def save_visualizations(model, sample, initial_kernels, device):
    """Save the MNIST sample, its first-layer maps, and kernel weights."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image, label = sample
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"MNIST digit: {label}")
    plt.axis("off")
    plt.savefig(OUTPUT_DIR / "mnist_sample.png", bbox_inches="tight")
    plt.close()

    model.eval()
    with torch.no_grad():
        features = model.relu(model.conv1(image.unsqueeze(0).to(device))).cpu()[0]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.imshow(features[index], cmap="gray")
        axis.set_title(f"Map {index + 1}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_maps.png")
    plt.close(fig)

    before = initial_kernels[:, 0]
    after = model.conv1.weight.detach().cpu()[:, 0]
    limit = max(before.abs().max().item(), after.abs().max().item())
    fig, axes = plt.subplots(2, 16, figsize=(16, 3))
    for index in range(16):
        axes[0, index].imshow(before[index], cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[1, index].imshow(after[index], cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[0, index].set_title(str(index + 1))
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("Before")
    axes[1, 0].set_ylabel("After")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kernels_before_after.png")
    plt.close(fig)
# --8<-- [end:visualize]


def main():
    """Train, evaluate, and save the example MNIST CNN."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    test_data, train_loader, validation_loader, test_loader = make_loaders()
    assert len(train_loader.dataset) == TRAIN_SIZE

    model = SimpleCNN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 20490
    print("Trainable parameters:", parameter_count)
    inspect_shapes(model, device)

    initial_kernels = model.conv1.weight.detach().cpu().clone()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_loss, train_accuracy = evaluate(model, train_loader, loss_fn, device)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_fn, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"train loss = {train_loss:.4f}, train accuracy = {train_accuracy:.2f}%, "
            f"validation loss = {validation_loss:.4f}, validation accuracy = {validation_accuracy:.2f}%"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2f}%")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / "mnist_cnn.pth"
    torch.save(model.state_dict(), checkpoint)
    save_training_history(history)
    save_training_curves(history)
    save_visualizations(model, test_data[0], initial_kernels, device)
    print("Saved model and visualizations to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
