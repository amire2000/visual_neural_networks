"""Export the trained MNIST MLP and compare ONNX Runtime with PyTorch."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_mnist_mlp import IMAGE_SIZE, INPUT_CHANNELS, MNISTMLP, OUTPUT_DIR

INPUT_NAME = "images"
OUTPUT_NAME = "digit_scores"
EXPORT_BATCH_SIZE = 1
CHECK_BATCH_SIZE = 7
ATOLERANCE = 1e-5


def main():
    """Export a dynamic-batch ONNX model and check its runtime outputs."""
    checkpoint_path = OUTPUT_DIR / "best_mlp.pth"
    onnx_path = OUTPUT_DIR / "mnist_mlp.onnx"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run train_mnist_mlp.py first."
        )

    model = MNISTMLP().eval()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    example_images = torch.zeros(
        EXPORT_BATCH_SIZE, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )
    # --8<-- [start:export]
    batch_dimension = torch.export.Dim("batch")
    onnx_program = torch.onnx.export(
        model,
        (example_images,),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: batch_dimension},),
        dynamo=True,
    )
    # Save the exporter program after marking only the leading batch axis dynamic.
    onnx_program.save(str(onnx_path))
    # --8<-- [end:export]
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    check_images = torch.rand(
        CHECK_BATCH_SIZE, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )
    with torch.inference_mode():
        pytorch_scores = model(check_images).numpy()
    onnx_scores = session.run(
        [OUTPUT_NAME], {INPUT_NAME: check_images.numpy()}
    )[0]
    # Confirm exported scores remain numerically aligned with the PyTorch model.
    np.testing.assert_allclose(
        onnx_scores, pytorch_scores, rtol=ATOLERANCE, atol=ATOLERANCE
    )
    print("Exported:", onnx_path)
    print("Dynamic batch check:", onnx_scores.shape)
    print("ONNX Runtime matches PyTorch within tolerance.")


if __name__ == "__main__":
    main()
