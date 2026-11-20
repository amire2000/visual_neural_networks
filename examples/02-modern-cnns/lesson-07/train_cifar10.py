"""Train and inspect a small convolutional network on CIFAR-10."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

IMAGE_SIZE = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
CONV3_CHANNELS = 128
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2
BATCH_SIZE = 128
EPOCHS = 5
TRAIN_SIZE = 45_000
VALIDATION_SIZE = 5_000
LEARNING_RATE = 0.001
CROP_PADDING = 4
HORIZONTAL_FLIP_PROBABILITY = 0.5
SEED = 0
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
# --8<-- [start:augmentations]
TRAIN_TRANSFORM = transforms.Compose(
    (
        transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PADDING),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
        transforms.ToTensor(),
    )
)
EVALUATION_TRANSFORM = transforms.ToTensor()
# --8<-- [end:augmentations]


# --8<-- [start:model]
class CIFAR10CNN(nn.Module):
    """Classify 32 by 32 RGB images into CIFAR-10's ten categories."""

    def __init__(self):
        """Create three convolution and pooling blocks and a classifier."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CONV1_CHANNELS, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
            nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
            nn.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV3_CHANNELS * 4 * 4, NUM_CLASSES),
        )

    def forward(self, x):
        """Return ten raw class scores for each image in the batch."""
        return self.classifier(self.features(x))
# --8<-- [end:model]


# --8<-- [start:dataset]
def make_loaders():
    """Download CIFAR-10 and augment training images only."""
    augmented_data = datasets.CIFAR10(
        DATA_DIR, train=True, download=True, transform=TRAIN_TRANSFORM
    )
    evaluation_data = datasets.CIFAR10(
        DATA_DIR, train=True, download=True, transform=EVALUATION_TRANSFORM
    )
    test_data = datasets.CIFAR10(
        DATA_DIR, train=False, download=True, transform=EVALUATION_TRANSFORM
    )
    indices = torch.randperm(
        len(augmented_data), generator=torch.Generator().manual_seed(SEED)
    ).tolist()
    training_subset = Subset(augmented_data, indices[:TRAIN_SIZE])
    validation_subset = Subset(evaluation_data, indices[TRAIN_SIZE:])
    train_loader = DataLoader(training_subset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    return evaluation_data, train_loader, validation_loader, test_loader
# --8<-- [end:dataset]


def save_sample_grid(dataset):
    """Save the first ten CIFAR-10 training images with their class labels."""
    figure, axes = plt.subplots(2, 5, figsize=(12, 5))
    for index, axis in enumerate(axes.flat):
        image, label = dataset[index]
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(CLASS_NAMES[label])
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "sample_images.png", dpi=150)
    plt.close(figure)


def save_augmented_views(image):
    """Save one CIFAR-10 image beside three random training augmentations."""
    views = [EVALUATION_TRANSFORM(image)] + [TRAIN_TRANSFORM(image) for _ in range(3)]
    figure, axes = plt.subplots(1, len(views), figsize=(10, 2.8))
    titles = ("Original", "Random crop / flip", "Random crop / flip", "Random crop / flip")
    for axis, view, title in zip(axes, views, titles):
        axis.imshow(view.permute(1, 2, 0))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "augmentation_examples.png", dpi=150)
    plt.close(figure)


def inspect_shapes(model, device):
    """Print the tensor shape after every feature layer and the classifier."""
    model.eval()
    values = torch.zeros(1, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    print("Input:", list(values.shape))
    with torch.no_grad():
        for layer in model.features:
            values = layer(values)
            print(f"{layer.__class__.__name__}:", list(values.shape))
        values = model.classifier(values)
    print("Output:", list(values.shape))


def train_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100 * correct / total


def evaluate(model, loader, loss_fn, device):
    """Measure average loss and accuracy without changing model weights."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100 * correct / total


def save_first_layer_filters(model):
    """Save the first convolution's RGB weight slices as small color tiles."""
    weights = model.features[0].weight.detach().cpu()
    columns = 8
    rows = (weights.size(0) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, rows * 1.5))
    for index, axis in enumerate(axes.flat):
        if index < weights.size(0):
            kernel = weights[index].permute(1, 2, 0)
            kernel = kernel - kernel.min()
            kernel = kernel / (kernel.max() + 1e-8)
            axis.imshow(kernel)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rgb_filters.png", dpi=150)
    plt.close(figure)


# --8<-- [start:training]
def main():
    """Train the CIFAR-10 CNN, report metrics, and save visual outputs."""
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    training_data, train_loader, validation_loader, test_loader = make_loaders()
    sample, sample_label = training_data[0]
    print("Image shape:", list(sample.shape), "label:", CLASS_NAMES[sample_label])
    save_sample_grid(training_data)
    save_augmented_views(Image.fromarray(training_data.data[0]))

    model = CIFAR10CNN().to(device)
    inspect_shapes(model, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_fn, device
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS}: "
            f"train loss={train_loss:.3f}, train acc={train_accuracy:.1f}%, "
            f"validation loss={validation_loss:.3f}, "
            f"validation acc={validation_accuracy:.1f}%"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    print(f"Final test: loss={test_loss:.3f}, accuracy={test_accuracy:.1f}%")
    save_first_layer_filters(model)
    torch.save(model.state_dict(), OUTPUT_DIR / "cifar10_cnn.pth")
    print("Saved sample_images.png, augmentation_examples.png, rgb_filters.png, and cifar10_cnn.pth")
# --8<-- [end:training]


if __name__ == "__main__":
    main()
