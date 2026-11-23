"""Create FP16 and calibrated static INT8 ONNX variants of MobileNetV2."""

import random
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import IMAGE_SIZE
from train import DATA_ROOT, OUTPUT_DIR, SEED

FLOAT_MODEL = OUTPUT_DIR / "mobilenetv2_fp32.onnx"
FP16_MODEL = OUTPUT_DIR / "mobilenetv2_fp16.onnx"
INT8_MODEL = OUTPUT_DIR / "mobilenetv2_int8.onnx"
CALIBRATION_SAMPLE_COUNT = 256
CALIBRATION_BATCH_SIZE = 16
INPUT_NAME = "images"


class GTSRBCalibrationReader(CalibrationDataReader):
    """Yield normalized image batches from a seeded sample of training data."""

    def __init__(self, dataset):
        """Prepare deterministic training images and a restartable batch iterator."""
        generator = random.Random(SEED)
        indices = list(range(len(dataset)))
        generator.shuffle(indices)
        selected = Subset(dataset, indices[:CALIBRATION_SAMPLE_COUNT])
        self.loader = DataLoader(selected, batch_size=CALIBRATION_BATCH_SIZE)
        self.rewind()

    def get_next(self):
        """Return the next named ONNX input batch or None when calibration ends."""
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None
        return {INPUT_NAME: images.numpy().astype(np.float32, copy=False)}

    def rewind(self):
        """Restart the calibration pass."""
        self.iterator = iter(self.loader)


def main():
    """Convert FP32 to FP16 and build a training-calibrated INT8 Q/DQ graph."""
    if not FLOAT_MODEL.exists():
        raise FileNotFoundError(f"Export MobileNetV2 first; missing {FLOAT_MODEL}")

    fp16_graph = float16.convert_float_to_float16(
        onnx.load(str(FLOAT_MODEL)), keep_io_types=True
    )
    onnx.save(fp16_graph, FP16_MODEL)
    onnx.checker.check_model(onnx.load(str(FP16_MODEL)))

    training_data = datasets.GTSRB(
        DATA_ROOT,
        split="train",
        download=True,
        transform=transforms.Compose(
            (transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor())
        ),
    )
    reader = GTSRBCalibrationReader(training_data)
    quantize_static(
        model_input=str(FLOAT_MODEL),
        model_output=str(INT8_MODEL),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(onnx.load(str(INT8_MODEL)))
    print("Saved FP16 graph:", FP16_MODEL)
    print("Saved calibrated INT8 Q/DQ graph:", INT8_MODEL)
    print(f"Calibration used {CALIBRATION_SAMPLE_COUNT} images from the training split.")


if __name__ == "__main__":
    main()
