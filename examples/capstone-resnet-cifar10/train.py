"""Train one CIFAR-10 model, save learning curves, and evaluate its test set."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet_cifar import IMAGE_SIZE, INPUT_CHANNELS, NUM_CLASSES, ResNet20, SmallCNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BATCH_SIZE = 128
EPOCHS = 5
TRAIN_SIZE = 45_000
VALIDATION_SIZE = 5_000
LEARNING_RATE = 0.01
RANDOM_SEED = 23
NUM_WORKERS = 0


def make_loaders():
    """Download CIFAR-10 and create repeatable training, validation, and test loaders."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    training_transform = transforms.Compose(
        [transforms.RandomCrop(IMAGE_SIZE, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    evaluation_transform = transforms.Compose([transforms.ToTensor(), normalize])
    training_data = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=training_transform)
    validation_data = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=evaluation_transform)
    indices = torch.randperm(len(training_data), generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_indices, validation_indices = indices[:TRAIN_SIZE], indices[TRAIN_SIZE:]
    train_set = torch.utils.data.Subset(training_data, train_indices)
    validation_set = torch.utils.data.Subset(validation_data, validation_indices)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=evaluation_transform)
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
        DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
        evaluation_transform,
    )


def evaluate(model, loader, loss_function, device):
    """Return average loss and accuracy without updating model weights."""
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += loss_function(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_total / total, 100.0 * correct / total


def save_sample_grid(transform):
    """Save a small CIFAR-10 image grid for visual inspection."""
    dataset = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform)
    class_names = dataset.classes
    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for class_id, axis in enumerate(axes.flat):
        image, _ = next((image, label) for image, label in dataset if label == class_id)
        image = image * torch.tensor((0.247, 0.243, 0.261))[:, None, None]
        image = image + torch.tensor((0.4914, 0.4822, 0.4465))[:, None, None]
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(class_names[class_id])
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cifar10_samples.png", dpi=150)
    plt.close(figure)


def save_curves(history):
    """Plot training and validation loss and accuracy over epochs."""
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Correct predictions (%)", ylim=(0, 100))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(figure)


def main():
    """Train the selected model, choose by validation accuracy, and report test accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet20", "baseline"), default="resnet20")
    args = parser.parse_args()
    torch.manual_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader, evaluation_transform = make_loaders()
    save_sample_grid(evaluation_transform)
    model = (ResNet20() if args.model == "resnet20" else SmallCNN()).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
    history = []
    best_validation_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / f"{args.model}_best.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_total = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, loss_function, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / total,
            "train_accuracy": 100.0 * correct / total,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{EPOCHS} | train loss {row['train_loss']:.3f} "
            f"| train acc {row['train_accuracy']:.1f}% | val acc {validation_accuracy:.1f}%"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    save_curves(history)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}% (loss {test_loss:.3f})")
    print(f"Saved sample grid, learning curves, and checkpoint in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
