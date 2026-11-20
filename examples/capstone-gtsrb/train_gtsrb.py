"""Train and visually evaluate a CNN on the GTSRB traffic-sign dataset."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CLASS_COUNT = 43
CNN_IMAGE_SIZE = 48
RESNET_IMAGE_SIZE = 224
INPUT_CHANNELS = 3
CNN_CHANNELS = (32, 64, 128)
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2
BATCH_SIZE = 64
EPOCHS = 5
VALIDATION_FRACTION = 0.15
LEARNING_RATE = 0.001
AFFINE_DEGREES = 8
AFFINE_TRANSLATION = (0.06, 0.06)
SEED = 7
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
MAX_ERROR_IMAGES = 15
CLASS_IDS = tuple(range(CLASS_COUNT))


def parse_args():
    """Parse the one supported model choice from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("cnn", "resnet18"),
        default="cnn",
        help="Use the custom CNN, or optionally fine-tune pretrained ResNet-18.",
    )
    return parser.parse_args()


def make_transforms(model_name):
    """Create training augmentation and deterministic evaluation preprocessing."""
    image_size = CNN_IMAGE_SIZE if model_name == "cnn" else RESNET_IMAGE_SIZE
    normalize = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    training_transform = transforms.Compose(
        (
            transforms.Resize((image_size, image_size)),
            # Keep sign direction intact: horizontal flips can change its meaning.
            transforms.RandomAffine(
                degrees=AFFINE_DEGREES,
                translate=AFFINE_TRANSLATION,
            ),
            transforms.ToTensor(),
            normalize,
        )
    )
    evaluation_transform = transforms.Compose(
        (
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        )
    )
    return training_transform, evaluation_transform


def make_grouped_split(dataset):
    """Split each class by recording-track prefix, keeping groups intact."""
    groups_by_class = defaultdict(lambda: defaultdict(list))
    # File prefixes identify related frames that should not cross the split.
    for index, (image_path, label) in enumerate(dataset._samples):
        path = Path(image_path)
        track_prefix = path.stem.split("_", maxsplit=1)[0]
        groups_by_class[label][(path.parent.name, track_prefix)].append(index)

    random_generator = random.Random(SEED)
    training_indices = []
    validation_indices = []
    validation_index_set = set()
    for label in CLASS_IDS:
        track_groups = list(groups_by_class[label].values())
        if len(track_groups) < 2:
            raise ValueError(f"Class {label} needs at least two recording tracks to split.")
        random_generator.shuffle(track_groups)
        track_groups.sort(key=len, reverse=True)
        target_validation_count = round(
            sum(map(len, track_groups)) * VALIDATION_FRACTION
        )
        validation_count = 0
        for group in track_groups[:-1]:
            if validation_count >= target_validation_count:
                break
            validation_indices.extend(group)
            validation_index_set.update(group)
            validation_count += len(group)
        for group in track_groups:
            if not any(index in validation_index_set for index in group):
                training_indices.extend(group)

    training_groups = {
        (Path(dataset._samples[index][0]).parent.name,
         Path(dataset._samples[index][0]).stem.split("_", maxsplit=1)[0])
        for index in training_indices
    }
    validation_groups = {
        (Path(dataset._samples[index][0]).parent.name,
         Path(dataset._samples[index][0]).stem.split("_", maxsplit=1)[0])
        for index in validation_indices
    }
    assert training_groups.isdisjoint(validation_groups)
    assert set(dataset._samples[index][1] for index in training_indices) == set(CLASS_IDS)
    assert set(dataset._samples[index][1] for index in validation_indices) == set(CLASS_IDS)
    return training_indices, validation_indices


def make_loaders(training_transform, evaluation_transform):
    """Download GTSRB and create track-aware train, validation, and test loaders."""
    augmented_data = datasets.GTSRB(
        DATA_ROOT, split="train", download=True, transform=training_transform
    )
    evaluation_data = datasets.GTSRB(
        DATA_ROOT, split="train", download=True, transform=evaluation_transform
    )
    test_data = datasets.GTSRB(
        DATA_ROOT, split="test", download=True, transform=evaluation_transform
    )
    training_indices, validation_indices = make_grouped_split(augmented_data)
    train_loader = DataLoader(
        Subset(augmented_data, training_indices), batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(evaluation_data, validation_indices), batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    print(
        f"Split sizes: train={len(training_indices):,}, "
        f"validation={len(validation_indices):,}, test={len(test_data):,}"
    )
    return evaluation_data, train_loader, validation_loader, test_loader


# --8<-- [start:model]
class TrafficSignCNN(nn.Module):
    """Classify resized RGB traffic-sign crops into 43 categories."""

    def __init__(self):
        """Create three convolution blocks and a global-average-pool head."""
        super().__init__()
        layers = []
        input_channels = INPUT_CHANNELS
        for output_channels in CNN_CHANNELS:
            layers.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        KERNEL_SIZE,
                        padding=PADDING,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(POOL_SIZE),
                )
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(CNN_CHANNELS[-1], CLASS_COUNT),
        )

    def forward(self, images):
        """Return one raw score for each sign class in every image."""
        return self.classifier(self.features(images))


def make_model(model_name, device):
    """Build the required custom CNN or optional pretrained ResNet-18."""
    if model_name == "cnn":
        return TrafficSignCNN().to(device)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, CLASS_COUNT)
    return model.to(device)
# --8<-- [end:model]


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Train or evaluate one full loader and return average loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def save_class_samples(dataset):
    """Save one representative training image from every class."""
    first_indices = {}
    for index, (_, label) in enumerate(dataset._samples):
        first_indices.setdefault(label, index)
    figure, axes = plt.subplots(5, 9, figsize=(13, 8))
    for label, axis in enumerate(axes.flat):
        if label < CLASS_COUNT:
            image, _ = dataset[first_indices[label]]
            image = image.permute(1, 2, 0) * torch.tensor(IMAGE_STD) + torch.tensor(IMAGE_MEAN)
            axis.imshow(image.clamp(0, 1))
            axis.set_title(str(label))
        axis.axis("off")
    figure.suptitle("GTSRB training samples — class ID above each sign")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_samples.png", dpi=150)
    plt.close(figure)


def save_training_curves(history):
    """Plot training and validation loss and accuracy by epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].legend()
    figure.suptitle("Learning from traffic-sign examples")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def save_test_diagnostics(model, loader, device):
    """Evaluate once and save metrics, confusion matrix, and representative errors."""
    confusion = torch.zeros(CLASS_COUNT, CLASS_COUNT, dtype=torch.int64)
    errors = []
    total_loss = 0.0
    total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            total_loss += nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item()
            predictions = logits.argmax(dim=1).cpu()
            total += labels.size(0)
            for image, label, prediction in zip(images, labels, predictions):
                confusion[label, prediction] += 1
                if label != prediction and len(errors) < MAX_ERROR_IMAGES:
                    errors.append((image.clone(), int(label), int(prediction)))

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(confusion.numpy(), cmap="magma")
    axis.set_title("Official test set: correct and confused classes")
    axis.set_xlabel("Predicted class ID")
    axis.set_ylabel("Correct class ID")
    axis.set_xticks(range(0, CLASS_COUNT, 5))
    axis.set_yticks(range(0, CLASS_COUNT, 5))
    figure.colorbar(image, ax=axis, label="Image count")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    rows = max(1, (len(errors) + 4) // 5)
    figure, axes = plt.subplots(rows, 5, figsize=(12, rows * 2.4), squeeze=False)
    for axis, (tensor, label, prediction) in zip(axes.flat, errors):
        tensor = tensor.permute(1, 2, 0) * torch.tensor(IMAGE_STD) + torch.tensor(IMAGE_MEAN)
        axis.imshow(tensor.clamp(0, 1))
        axis.set_title(f"pred {prediction} / true {label}")
        axis.axis("off")
    for axis in axes.flat[len(errors):]:
        axis.axis("off")
    figure.suptitle("Misclassified official test images")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_errors.png", dpi=150)
    plt.close(figure)
    return total_loss / total, 100.0 * confusion.diag().sum().item() / total


def main():
    """Train the selected model, choose by validation, and report test metrics."""
    arguments = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model:", arguments.model, "| Device:", device)

    training_transform, evaluation_transform = make_transforms(arguments.model)
    samples, train_loader, validation_loader, test_loader = make_loaders(
        training_transform, evaluation_transform
    )
    save_class_samples(samples)

    model = make_model(arguments.model, device)
    example_batch = torch.zeros(
        1,
        INPUT_CHANNELS,
        CNN_IMAGE_SIZE if arguments.model == "cnn" else RESNET_IMAGE_SIZE,
        CNN_IMAGE_SIZE if arguments.model == "cnn" else RESNET_IMAGE_SIZE,
        device=device,
    )
    model.eval()
    with torch.inference_mode():
        example_logits = model(example_batch)
    print("Example logits shape:", list(example_logits.shape))
    assert example_logits.shape == (1, CLASS_COUNT)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = defaultdict(list)
    best_validation_accuracy = -1.0
    checkpoint_path = OUTPUT_DIR / f"best_{arguments.model}.pth"
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, loss_fn, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch}/{EPOCHS}: train loss={train_loss:.3f}, "
            f"train acc={train_accuracy:.1f}%, validation loss={validation_loss:.3f}, "
            f"validation acc={validation_accuracy:.1f}%"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_accuracy = save_test_diagnostics(model, test_loader, device)
    print(
        f"Best validation accuracy={best_validation_accuracy:.1f}% | "
        f"Official test loss={test_loss:.3f}, accuracy={test_accuracy:.1f}%"
    )
    save_training_curves(history)
    print("Saved checkpoint and visual results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
