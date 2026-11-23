"""Save representative MNIST images for TensorRT INT8 calibration."""

from pathlib import Path

import numpy as np

from infer_onnx import INPUT_NAME

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CALIBRATION_IMAGES = 256


def main():
    """Write normalized MNIST inputs under the ONNX input name."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use representative training images for calibration; the test batch is only a small demo.
    from torchvision import datasets, transforms

    data = datasets.MNIST(
        OUTPUT_DIR.parents[2] / "data" / "mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    indices = np.random.default_rng(31).choice(len(data), CALIBRATION_IMAGES, replace=False)
    calibration = np.stack([data[int(index)][0].numpy() for index in indices])
    assert calibration.shape == (CALIBRATION_IMAGES, 1, 28, 28)
    np.savez(OUTPUT_DIR / "mnist_calibration.npz", **{INPUT_NAME: calibration})
    print("Saved calibration inputs:", calibration.shape)


if __name__ == "__main__":
    main()
