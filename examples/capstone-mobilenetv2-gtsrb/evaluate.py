"""Compare the trained baseline and MobileNetV2 on GTSRB's official test split."""

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_COUNT, IMAGE_SIZE, make_model
from train import BATCH_SIZE, DATA_ROOT, MODEL_NAMES, OUTPUT_DIR

WARMUP_RUNS = 5
TIMED_RUNS = 30
VISUALIZATION_COUNT = 16


def load_test_data():
    """Load the untouched official test split with deterministic preprocessing."""
    dataset = datasets.GTSRB(
        DATA_ROOT,
        split="test",
        download=True,
        transform=transforms.Compose(
            (transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor())
        ),
    )
    return dataset, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def synchronize(device):
    """Wait for queued CUDA work so measured wall time includes model execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(model, images, device):
    """Return median one-batch inference latency after warmup."""
    model.eval()
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(images)
        synchronize(device)
        durations = []
        for _ in range(TIMED_RUNS):
            synchronize(device)
            start = perf_counter()
            model(images)
            synchronize(device)
            durations.append(1000.0 * (perf_counter() - start))
    return float(np.median(durations))


def evaluate_one(model_name, dataset, loader, device):
    """Load one validation-selected model and calculate test and size metrics."""
    checkpoint = OUTPUT_DIR / f"{model_name}_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Train {model_name} first; missing {checkpoint}")
    model = make_model(model_name).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    correct = 0
    predictions = []
    labels_seen = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            batch_predictions = logits.argmax(dim=1).cpu()
            predictions.extend(batch_predictions.tolist())
            labels_seen.extend(labels.tolist())
            correct += (batch_predictions == labels).sum().item()
    sample_images, _ = dataset[0]
    latency_images = torch.stack(
        [dataset[index][0] for index in range(min(BATCH_SIZE, len(dataset)))]
    ).to(device)
    latency_ms = measure_latency(model, latency_images, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model": model_name,
        "test_accuracy_percent": 100.0 * correct / len(dataset),
        "parameters": parameter_count,
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024 * 1024),
        "batch_size": len(latency_images),
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_image": latency_ms / len(latency_images),
        "predictions": predictions,
        "labels": labels_seen,
        "example_shape": list(sample_images.shape),
    }


def save_comparison(results, dataset):
    """Save a compact plot of test accuracy and model size for both networks."""
    names = [result["model"] for result in results]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(names, [result["test_accuracy_percent"] for result in results])
    axes[0].set_title("Official test accuracy")
    axes[0].set_ylabel("Percent")
    axes[1].bar(names, [result["checkpoint_size_mb"] for result in results])
    axes[1].set_title("Checkpoint size")
    axes[1].set_ylabel("MiB")
    axes[2].bar(names, [result["latency_ms_per_image"] for result in results])
    axes[2].set_title("PyTorch latency")
    axes[2].set_ylabel("Median ms/image")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        image, label = dataset[index]
        axis.imshow(image.permute(1, 2, 0))
        # Show predictions from each model so errors can be inspected visually.
        captions = [f"{result['model']}: {result['predictions'][index]}" for result in results]
        axis.set_title(f"true {label}\n" + " · ".join(captions), fontsize=7)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150)
    plt.close(figure)


def main():
    """Evaluate both models once on test data and print a comparable summary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    dataset, loader = load_test_data()
    results = [evaluate_one(name, dataset, loader, device) for name in MODEL_NAMES]
    public_results = [
        {key: value for key, value in result.items() if key not in ("predictions", "labels", "example_shape")}
        for result in results
    ]
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(public_results, indent=2) + "\n")
    save_comparison(results, dataset)
    print(f"Device: {device} · GTSRB official test images: {len(dataset)}")
    print("| Model | Test accuracy | Parameters | Checkpoint MiB | Median ms/image |")
    print("|---|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['model']} | {result['test_accuracy_percent']:.2f}% | "
            f"{result['parameters']:,} | {result['checkpoint_size_mb']:.2f} | "
            f"{result['latency_ms_per_image']:.3f} |"
        )
    print("Saved comparison.json, model_comparison.png, and test_predictions.png")


if __name__ == "__main__":
    main()
