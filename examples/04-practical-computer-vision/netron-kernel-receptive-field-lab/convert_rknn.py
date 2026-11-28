"""Optional RKNN conversion; requires compatible RKNN-Toolkit2 on the host."""

from pathlib import Path

from rknn.api import RKNN

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAMES = ("baseline", "optimized")
TARGET_PLATFORM = "rk3566"


def convert(model_name):
    """Convert one ONNX model without quantization and release the toolkit object."""
    converter = RKNN(verbose=True)
    try:
        if converter.config(target_platform=TARGET_PLATFORM) != 0:
            raise RuntimeError("RKNN config failed")
        if converter.load_onnx(model=str(OUTPUT_DIR / f"{model_name}.onnx")) != 0:
            raise RuntimeError(f"Could not load {model_name}.onnx")
        if converter.build(do_quantization=False) != 0:
            raise RuntimeError(f"Could not build {model_name} RKNN model")
        output_path = OUTPUT_DIR / f"{model_name}.rknn"
        if converter.export_rknn(str(output_path)) != 0:
            raise RuntimeError(f"Could not export {output_path}")
        print("Exported:", output_path)
    finally:
        converter.release()


def main():
    """Convert both graphs after their ONNX files have been generated."""
    for model_name in MODEL_NAMES:
        model_path = OUTPUT_DIR / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Export both models first; missing {model_path}")
        convert(model_name)


if __name__ == "__main__":
    main()
