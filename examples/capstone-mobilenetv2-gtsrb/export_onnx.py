"""Export the selected MobileNetV2 checkpoint and compare ONNX Runtime logits."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import IMAGE_SIZE, MobileNetV2Small
from train import OUTPUT_DIR

INPUT_NAME = "images"
OUTPUT_NAME = "class_scores"
EXPORT_BATCH_SIZE = 1
CHECK_BATCH_SIZE = 3
ATOLERANCE = 1e-4


def main():
    """Export a dynamic-batch graph, validate it, and check runtime agreement."""
    checkpoint_path = OUTPUT_DIR / "mobilenetv2_best.pth"
    onnx_path = OUTPUT_DIR / "mobilenetv2_fp32.onnx"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Train MobileNetV2 first; missing {checkpoint_path}")
    model = MobileNetV2Small().eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    example = torch.zeros(EXPORT_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    batch_axis = torch.export.Dim("batch")
    exported = torch.onnx.export(
        model,
        (example,),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: batch_axis},),
        dynamo=True,
    )
    exported.save(str(onnx_path))
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = torch.rand(CHECK_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        torch_scores = model(sample).numpy()
    onnx_scores = session.run([OUTPUT_NAME], {INPUT_NAME: sample.numpy()})[0]
    np.testing.assert_allclose(onnx_scores, torch_scores, rtol=ATOLERANCE, atol=ATOLERANCE)
    print("Saved:", onnx_path)
    print("Dynamic-batch output:", onnx_scores.shape)
    print("ONNX Runtime CPU scores match PyTorch within tolerance.")


if __name__ == "__main__":
    main()
