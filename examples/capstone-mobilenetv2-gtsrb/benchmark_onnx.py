"""Compare ONNX model load time, inference latency, and test accuracy by provider."""

from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMAGE_SIZE
from train import BATCH_SIZE, DATA_ROOT, OUTPUT_DIR

WARMUP_RUNS = 5
TIMED_RUNS = 30
INPUT_NAME = "images"
OUTPUT_NAME = "class_scores"


def make_test_loader():
    """Load official GTSRB test images with the model's deterministic preprocessing."""
    dataset = datasets.GTSRB(
        DATA_ROOT,
        split="test",
        download=True,
        transform=transforms.Compose(
            (transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor())
        ),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_accuracy(session, loader):
    """Return official test accuracy using one ONNX Runtime session."""
    correct = 0
    total = 0
    for images, labels in loader:
        scores = session.run([OUTPUT_NAME], {INPUT_NAME: images.numpy()})[0]
        correct += int((scores.argmax(axis=1) == labels.numpy()).sum())
        total += len(labels)
    return 100.0 * correct / total, total


def measure_latency(session, sample):
    """Return median runtime for one preprocessed batch after warmup."""
    feed = {INPUT_NAME: sample}
    for _ in range(WARMUP_RUNS):
        session.run([OUTPUT_NAME], feed)
    durations = []
    for _ in range(TIMED_RUNS):
        start = perf_counter()
        session.run([OUTPUT_NAME], feed)
        durations.append(1000.0 * (perf_counter() - start))
    return float(np.median(durations))


def try_benchmark(model_path, precision, provider, loader):
    """Measure one model/provider pair or report why it cannot run."""
    if not model_path.exists():
        print(f"SKIP {precision} {provider}: missing {model_path.name}")
        return
    if precision == "FP16" and provider == "CPUExecutionProvider":
        print("SKIP FP16 CPUExecutionProvider: ONNX Runtime CPU does not support FP16 ops")
        return
    start = perf_counter()
    try:
        session = ort.InferenceSession(str(model_path), providers=[provider])
        load_ms = 1000.0 * (perf_counter() - start)
        first_images, _ = next(iter(loader))
        sample = first_images.numpy()
        latency_ms = measure_latency(session, sample)
        accuracy, image_count = evaluate_accuracy(session, loader)
    except Exception as error:
        print(f"SKIP {precision} {provider}: {type(error).__name__}: {error}")
        return
    throughput = len(sample) * 1000.0 / latency_ms
    print(
        f"| {precision} | {provider} | {load_ms:.1f} | {latency_ms:.3f} | "
        f"{latency_ms / len(sample):.3f} | {throughput:.1f} | "
        f"{accuracy:.2f}% ({image_count}) | {model_path.stat().st_size / (1024 * 1024):.2f} |"
    )


def main():
    """Run supported FP32/FP16/INT8 variants and explicitly mark unavailable pairs."""
    loader = make_test_loader()
    available = set(ort.get_available_providers())
    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"
    print("Accuracy uses the full official GTSRB test split.")
    print("Model inputs are preprocessed once; host-to-device transfers are included by ORT.")
    print("| Precision | Provider | Load ms | Batch ms | ms/image | Images/s | Test accuracy | Model MiB |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    variants = (
        ("FP32", OUTPUT_DIR / "mobilenetv2_fp32.onnx", (cpu, cuda)),
        ("FP16", OUTPUT_DIR / "mobilenetv2_fp16.onnx", (cuda,)),
        ("INT8", OUTPUT_DIR / "mobilenetv2_int8.onnx", (cpu, cuda)),
    )
    for precision, path, providers in variants:
        for provider in providers:
            if provider not in available:
                print(f"SKIP {precision} {provider}: provider not installed")
                continue
            try_benchmark(path, precision, provider, loader)
    if cuda not in available:
        print("CUDA comparisons require a compatible onnxruntime-gpu and CUDA/cuDNN setup.")
    print("INT8 CUDA execution is attempted only when CUDAExecutionProvider is available.")


if __name__ == "__main__":
    main()
