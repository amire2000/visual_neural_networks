"""Classify and visualize a shuffled batch of MNIST test digits with ONNX Runtime."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FLOAT_ONNX_PATH = OUTPUT_DIR / "mnist_mlp.onnx"
INT8_ONNX_PATH = OUTPUT_DIR / "mnist_mlp_int8.onnx"

IMAGE_SIZE = 28
INFERENCE_BATCH_SIZE = 16
GRID_COLUMNS = 4
GRID_ROWS = 4
SEED = 31
INPUT_NAME = "images"
OUTPUT_NAME = "digit_scores"
QUANTIZED_RESULT_NAME = "onnx_int8_predictions.png"
FLOAT_RESULT_NAME = "onnx_predictions.png"


def parse_args():
    """Read whether inference should use the int8 ONNX model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="run the int8 ONNX model instead of the float model",
    )
    return parser.parse_args()


def make_test_batch():
    """Load a reproducible shuffled batch from the official MNIST test split."""
    # ToTensor converts grayscale pixels to float32 in [0, 1] with shape [1, 28, 28].
    test_data = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    # The labels are kept for evaluation and figure titles, not passed to ONNX.
    return next(iter(test_loader))


def main():
    """Run ONNX inference on a test batch and save a labeled prediction grid."""
    args = parse_args()
    onnx_path = INT8_ONNX_PATH if args.quantized else FLOAT_ONNX_PATH
    result_name = QUANTIZED_RESULT_NAME if args.quantized else FLOAT_RESULT_NAME
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Missing {onnx_path}. Run the matching export or quantization step first."
        )
    images, labels = make_test_batch()
    # --8<-- [start:inference]
    # Create a CPU session for the ONNX graph, not the original PyTorch model.
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    # Convert the float32 image batch to NumPy for ONNX Runtime's named input.
    scores = session.run(
        [OUTPUT_NAME], {INPUT_NAME: images.numpy()}
    )[0]
    # Argmax works directly on logits; softmax is only needed for probabilities.
    predictions = scores.argmax(axis=1)
    # Stable softmax: subtract each row's largest score before exponentiating.
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    confidences = probabilities / probabilities.sum(axis=1, keepdims=True)
    # Labels are used here only to measure this sample batch's accuracy.
    batch_accuracy = 100.0 * (predictions == labels.numpy()).mean()
    # --8<-- [end:inference]

    # Show each prediction beside its true test label, with color marking correctness.
    figure, axes = plt.subplots(
        GRID_ROWS, GRID_COLUMNS, figsize=(9, 9), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        is_correct = predictions[index] == labels[index].item()
        axis.imshow(images[index, 0], cmap="gray")
        axis.set_title(
            f"true {labels[index].item()} · pred {predictions[index]} "
            f"({confidences[index, predictions[index]]:.0%})",
            color="#15803d" if is_correct else "#dc2626",
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle(f"MNIST digits predicted by ONNX Runtime · {batch_accuracy:.1f}% batch accuracy")
    figure.tight_layout()
    result_path = OUTPUT_DIR / result_name
    # Save the visual result before opening it in a plotting window.
    figure.savefig(result_path, dpi=160, bbox_inches="tight")
    print("Batch size:", len(images))
    print("Model:", onnx_path.name)
    print("Batch accuracy:", f"{batch_accuracy:.2f}%")
    print("Saved prediction grid:", result_path)
    plt.show()


if __name__ == "__main__":
    main()
