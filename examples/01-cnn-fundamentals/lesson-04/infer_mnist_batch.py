# --8<-- [start:inference_example]
"""Run and visualize predictions for a random batch of MNIST test images."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_mnist import DATA_DIR, OUTPUT_DIR, SimpleCNN

INFERENCE_BATCH_SIZE = 16
SEED = 42
CLASS_NAMES = [str(digit) for digit in range(10)]
CHECKPOINT = OUTPUT_DIR / "mnist_cnn.pth"


def main():
    """Load the trained CNN, predict a shuffled test batch, and plot results."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT} not found. Run train_mnist.py before this example."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    random_batch = DataLoader(
        test_data,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    images, labels = next(iter(random_batch))

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        probabilities = model(images.to(device)).softmax(dim=1).cpu()

    confidence, predictions = probabilities.max(dim=1)
    columns = 4
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(9, 2.5 * rows))
    for index, axis in enumerate(axes.flat):
        if index < len(images):
            predicted = predictions[index].item()
            actual = labels[index].item()
            color = "#15803d" if predicted == actual else "#dc2626"
            axis.imshow(images[index].squeeze(), cmap="gray")
            axis.set_title(
                f"Predicted: {CLASS_NAMES[predicted]} ({confidence[index]:.0%})\n"
                f"Actual: {CLASS_NAMES[actual]}",
                color=color,
                fontsize=9,
            )
        axis.axis("off")

    figure.suptitle("Random MNIST test batch · green = correct, red = incorrect")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "random_inference_batch.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)

    correct = (predictions == labels).sum().item()
    print(f"Correct predictions in this batch: {correct}/{len(labels)}")
    print("Saved visualization to:", output_path)
# --8<-- [end:inference_example]


if __name__ == "__main__":
    main()
