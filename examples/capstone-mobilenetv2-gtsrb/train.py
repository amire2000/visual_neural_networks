"""Train a small CNN or MobileNetV2 on the German traffic-sign dataset."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import IMAGE_SIZE, make_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "gtsrb"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAMES = ("baseline", "mobilenetv2")
CLASS_COUNT = 43
BATCH_SIZE = 64
DEFAULT_EPOCHS = 8
VALIDATION_FRACTION = 0.15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
AFFINE_DEGREES = 8
AFFINE_TRANSLATION = (0.06, 0.06)
SEED = 7
CLASS_IDS = tuple(range(CLASS_COUNT))


def parse_args():
    """Read the architecture and training duration from command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()


def make_transforms():
    """Create shared 96-pixel preprocessing with sign-safe training augmentation."""
    fixed = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    training = transforms.Compose(
        (
            fixed,
            # Avoid flips because horizontally reversed road signs can change meaning.
            transforms.RandomAffine(AFFINE_DEGREES, translate=AFFINE_TRANSLATION),
            transforms.ToTensor(),
        )
    )
    evaluation = transforms.Compose((fixed, transforms.ToTensor()))
    return training, evaluation


def make_grouped_split(dataset):
    """Split each class by recording-track prefix to reduce near-duplicate leakage."""
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for index, (image_path, label) in enumerate(dataset._samples):
        path = Path(image_path)
        group = (path.parent.name, path.stem.split("_", maxsplit=1)[0])
        groups_by_class[label][group].append(index)

    generator = random.Random(SEED)
    training_indices = []
    validation_indices = []
    for label in CLASS_IDS:
        groups = list(groups_by_class[label].values())
        if len(groups) < 2:
            raise ValueError(f"Class {label} needs at least two recording tracks.")
        generator.shuffle(groups)
        groups.sort(key=len, reverse=True)
        target_count = round(sum(map(len, groups)) * VALIDATION_FRACTION)
        validation_count = 0
        validation_groups = []
        for group in groups[:-1]:
            if validation_count >= target_count:
                break
            validation_groups.append(group)
            validation_count += len(group)
        validation_indices.extend(index for group in validation_groups for index in group)
        validation_set = {index for group in validation_groups for index in group}
        training_indices.extend(
            index for group in groups for index in group if index not in validation_set
        )

    def recording_groups(indices):
        """Return recording identifiers for the requested dataset indices."""
        return {
            (
                Path(dataset._samples[index][0]).parent.name,
                Path(dataset._samples[index][0]).stem.split("_", maxsplit=1)[0],
            )
            for index in indices
        }

    assert recording_groups(training_indices).isdisjoint(recording_groups(validation_indices))
    assert {dataset._samples[i][1] for i in training_indices} == set(CLASS_IDS)
    assert {dataset._samples[i][1] for i in validation_indices} == set(CLASS_IDS)
    return training_indices, validation_indices


def make_loaders():
    """Download GTSRB and make track-aware train and validation loaders."""
    training_transform, evaluation_transform = make_transforms()
    training_data = datasets.GTSRB(
        DATA_ROOT, split="train", download=True, transform=training_transform
    )
    evaluation_data = datasets.GTSRB(
        DATA_ROOT, split="train", download=True, transform=evaluation_transform
    )
    train_indices, validation_indices = make_grouped_split(training_data)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(training_data, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(evaluation_data, validation_indices), batch_size=BATCH_SIZE
    )
    return train_loader, validation_loader


def run_epoch(model, loader, loss_function, device, optimizer=None):
    """Train or validate for one epoch and return mean loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def save_curves(history, model_name):
    """Save loss and accuracy curves for this model's training run."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.suptitle(f"GTSRB training: {model_name}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{model_name}_training.png", dpi=150)
    plt.close(figure)


def main():
    """Train one model and save its best validation checkpoint and learning curves."""
    arguments = parse_args()
    if arguments.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = make_loaders()
    model = make_model(arguments.model).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = {f"{split}_{metric}": [] for split in ("train", "validation") for metric in ("loss", "accuracy")}
    best_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / f"{arguments.model}_best.pth"
    print(f"Model: {arguments.model} · device: {device} · parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, arguments.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_function, device
        )
        for name, value in (
            ("train_loss", train_loss),
            ("train_accuracy", train_accuracy),
            ("validation_loss", validation_loss),
            ("validation_accuracy", validation_accuracy),
        ):
            history[name].append(value)
        print(
            f"Epoch {epoch}/{arguments.epochs}: train {train_accuracy:.2f}% · "
            f"validation {validation_accuracy:.2f}%"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)
    save_curves(history, arguments.model)
    print("Saved best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
