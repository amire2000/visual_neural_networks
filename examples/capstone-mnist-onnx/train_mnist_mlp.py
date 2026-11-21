"""Train a fully connected multilayer perceptron on MNIST."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Resolve data and outputs from the repository, so commands work from its root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# --8<-- [start:model]
from torch import nn

# Named shape and dropout settings make the network dimensions easy to inspect.
IMAGE_SIZE = 28
INPUT_CHANNELS = 1
INPUT_FEATURES = INPUT_CHANNELS * IMAGE_SIZE * IMAGE_SIZE
HIDDEN_UNITS = (256, 128)
NUM_CLASSES = 10
DROPOUT_RATE = 0.2


class MNISTMLP(nn.Module):
    """Classify flattened 28 by 28 grayscale images into ten digits."""

    def __init__(self):
        """Create two hidden layers and a ten-score output layer."""
        super().__init__()
        # Flatten keeps the batch axis and turns each image into 784 pixel values.
        # Each Linear layer maps the last dimension to the next number of units.
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, HIDDEN_UNITS[0]),
            nn.ReLU(),
            # Dropout is active in training mode and disabled for validation/inference.
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_UNITS[0], HIDDEN_UNITS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS[1], NUM_CLASSES),
        )

    def forward(self, images):
        """Return one digit score for each class and each input image."""
        return self.layers(images)
# --8<-- [end:model]

TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
SEED = 21


# --8<-- [start:loaders]
def make_loaders():
    """Download MNIST and create deterministic training and validation splits."""
    # ToTensor converts pixels to float tensors with shape [1, 28, 28] and values [0, 1].
    dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    training_data, validation_data = random_split(
        dataset,
        (TRAIN_SIZE, VALIDATION_SIZE),
        generator=torch.Generator().manual_seed(SEED),
    )
    # Shuffle training examples each epoch; keep validation order stable.
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    return train_loader, validation_loader
# --8<-- [end:loaders]


# --8<-- [start:run_epoch]
def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Train or evaluate one pass and return mean loss and accuracy."""
    # An optimizer means training; without one, this pass only measures performance.
    is_training = optimizer is not None
    # This also switches Dropout on for training and off for validation.
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_training:
            # Clear gradients left from the previous batch before computing new ones.
            optimizer.zero_grad()
        # Validation needs predictions and loss, but it does not need gradients.
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            # CrossEntropyLoss consumes raw class scores and integer digit labels.
            loss = loss_fn(logits, labels)
            if is_training:
                # Compute gradients, then let Adam update the trainable parameters.
                loss.backward()
                optimizer.step()
        # Weight each batch mean by its size to get correct dataset-wide averages.
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total
# --8<-- [end:run_epoch]


def save_training_curves(history):
    """Save training and validation loss and accuracy by epoch."""
    # Plot both metrics so learning and train/validation gaps are easy to inspect.
    epoch_numbers = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epoch_numbers, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epoch_numbers, history["validation_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epoch_numbers, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epoch_numbers, history["validation_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].legend()
    figure.suptitle("MNIST MLP learning curves")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def main():
    """Train the MLP, save its best validation weights, and graph its learning."""
    # Seed initialization and the data split for a repeatable teaching run.
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use CUDA when available; CPU is the fallback.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders()
    model = MNISTMLP().to(device)
    # Cross entropy compares ten logits with the correct digit; Adam updates weights.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Keep one metric per epoch for the graph at the end of training.
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
    best_validation_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / "best_mlp.pth"

    print("Device:", device)
    print("Architecture:", INPUT_FEATURES, "→", HIDDEN_UNITS[0], "→", HIDDEN_UNITS[1], "→", NUM_CLASSES)
    # --8<-- [start:epoch_loop]
    # Each epoch trains once, then measures generalization on held-out validation data.
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_fn, device
        )
        # Keep the values needed to draw both learning curves after training.
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch}/{EPOCHS}: train loss={train_loss:.4f}, "
            f"train accuracy={train_accuracy:.2f}%, "
            f"validation loss={validation_loss:.4f}, "
            f"validation accuracy={validation_accuracy:.2f}%"
        )
        # Select weights with validation data, reserving the test set for inference.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)
    # --8<-- [end:epoch_loop]

    save_training_curves(history)
    print("Best validation accuracy:", f"{best_validation_accuracy:.2f}%")
    print("Saved weights:", checkpoint_path)
    print("Saved graph:", OUTPUT_DIR / "training_curves.png")


if __name__ == "__main__":
    main()
