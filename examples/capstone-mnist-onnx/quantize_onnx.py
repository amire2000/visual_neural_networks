"""Create a dynamic int8 copy of the exported MNIST MLP ONNX model."""

# --8<-- [start:quantization]

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FLOAT_MODEL_PATH = OUTPUT_DIR / "mnist_mlp.onnx"
INT8_MODEL_PATH = OUTPUT_DIR / "mnist_mlp_int8.onnx"


def main():
    """Quantize the exported model and validate the saved ONNX graph."""
    if not FLOAT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FLOAT_MODEL_PATH}. Run export_onnx.py first."
        )

    # Keep the float model so inference can compare both versions.
    quantize_dynamic(
        model_input=str(FLOAT_MODEL_PATH),
        model_output=str(INT8_MODEL_PATH),
        weight_type=QuantType.QInt8,
    )
    onnx.checker.check_model(onnx.load(str(INT8_MODEL_PATH)))
    print("Saved quantized model:", INT8_MODEL_PATH)


if __name__ == "__main__":
    main()
# --8<-- [end:quantization]
