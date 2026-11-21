"""Train an MNIST CNN while logging metrics and images to TensorBoard."""

import argparse
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "tensorboard"
BATCH_SIZE = 64
TRAIN_SIZE = 50000
VALIDATION_SIZE = 10000
EPOCHS = 20
SEED = 42


class SimpleCNN(nn.Module):
    """Classify 28 by 28 grayscale images into ten digit classes."""

    def __init__(self):
        """Create the same convolutional model used in Lessons 3–5."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, images):
        """Return ten raw class scores for each input image."""
        images = self.pool(self.relu(self.conv1(images)))
        images = self.pool(self.relu(self.conv2(images)))
        return self.fc(self.flatten(images))


def make_loaders():
    """Load MNIST and split its original training set into train and validation data."""
    dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    train_dataset, validation_dataset = random_split(
        dataset,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return validation_loader, train_loader


def evaluate(model, loader, loss_fn, device):
    """Return sample-weighted mean loss and accuracy without changing weights."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_size = labels.size(0)
            loss_sum += loss_fn(logits, labels).item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_size
    return loss_sum / count, 100.0 * correct / count


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch and return sample-weighted loss and online accuracy."""
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size
    return loss_sum / count, 100.0 * correct / count


def parse_args():
    """Read the learning rate and a safe TensorBoard run name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be a finite positive number")
    args.run_name = args.run_name or f"mnist_lr_{args.learning_rate:g}".replace(".", "p")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("--run-name may contain only letters, numbers, dots, underscores, and hyphens")
    return args


def main():
    """Train one independent run, log its metrics, and save its best checkpoint."""
    args = parse_args()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_loader, train_loader = make_loaders()
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = OUTPUT_DIR / "runs" / args.run_name
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.run_name}.pth"
    # A run-specific event folder lets TensorBoard compare separate experiments.
    writer = SummaryWriter(log_dir=str(run_dir))
    best_validation_loss = float("inf")

    try:
        sample_images, _ = next(iter(validation_loader))
        writer.add_image(
            "MNIST/Validation examples",
            make_grid(sample_images[:16], nrow=4, normalize=True),
            global_step=0,
        )

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device
            )
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, loss_fn, device
            )
            # --8<-- [start:logging]
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", validation_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/Validation", validation_accuracy, epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)

            if epoch % 5 == 0:
                writer.add_histogram("Weights/conv1", model.conv1.weight.detach().cpu(), epoch)
                writer.add_histogram("Weights/conv2", model.conv2.weight.detach().cpu(), epoch)
            # Make this epoch's event records available to an open dashboard.
            writer.flush()
            # --8<-- [end:logging]

            # --8<-- [start:checkpoint]
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), checkpoint_path)
            # --8<-- [end:checkpoint]

            print(
                f"Epoch {epoch:02d}/{EPOCHS}: train loss={train_loss:.4f}, "
                f"validation loss={validation_loss:.4f}, "
                f"train accuracy={train_accuracy:.2f}%, "
                f"validation accuracy={validation_accuracy:.2f}%"
            )
    finally:
        writer.close()

    print("TensorBoard run:", run_dir)
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
