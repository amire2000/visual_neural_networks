"""Export baseline and depthwise-separable CNNs for Netron inspection."""

from pathlib import Path

import torch
import onnx
from torch import nn

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
INPUT_CHANNELS = 3
STEM_CHANNELS = 32
TARGET_CHANNELS = 64
CLASS_COUNT = 10
INPUT_SIZE = 224
STEM_STRIDE = 2
STEM_KERNEL_SIZE = 3
STEM_PADDING = 1
TARGET_KERNEL_SIZE = 3
TARGET_PADDING = 1
POINTWISE_KERNEL_SIZE = 1
POOL_OUTPUT_SIZE = 1
ONNX_OPSET = 17
WARMUP_BATCH = 1
TARGET_SPATIAL_SIZE = INPUT_SIZE // STEM_STRIDE
EXPECTED_ORIGINAL_TARGET_MACS = 231_211_008
EXPECTED_OPTIMIZED_TARGET_MACS = 29_302_784


# --8<-- [start:model]
class SimpleCNN(nn.Module):
    """Classify one RGB image with a standard or depthwise-separable layer."""

    def __init__(self, optimized=False):
        """Build the baseline graph or replace its target convolution."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                STEM_CHANNELS,
                STEM_KERNEL_SIZE,
                stride=STEM_STRIDE,
                padding=STEM_PADDING,
            ),
            nn.ReLU(),
        )
        if optimized:
            # Filter each channel spatially, then mix channels with a 1×1 convolution.
            self.conv = nn.Sequential(
                nn.Conv2d(
                    STEM_CHANNELS,
                    STEM_CHANNELS,
                    TARGET_KERNEL_SIZE,
                    padding=TARGET_PADDING,
                    groups=STEM_CHANNELS,
                    bias=False,
                ),
                nn.Conv2d(STEM_CHANNELS, TARGET_CHANNELS, POINTWISE_KERNEL_SIZE),
            )
        else:
            self.conv = nn.Conv2d(
                STEM_CHANNELS,
                TARGET_CHANNELS,
                TARGET_KERNEL_SIZE,
                padding=TARGET_PADDING,
            )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(POOL_OUTPUT_SIZE)
        self.classifier = nn.Linear(TARGET_CHANNELS, CLASS_COUNT)

    def forward(self, images):
        """Return one class-score vector for every image in the batch."""
        features = self.stem(images)
        features = self.relu(self.conv(features))
        features = self.pool(features)
        return self.classifier(torch.flatten(features, 1))
# --8<-- [end:model]


def target_layer_metrics(optimized):
    """Calculate target-layer parameters and MACs for one graph variant."""
    spatial_positions = TARGET_SPATIAL_SIZE * TARGET_SPATIAL_SIZE
    kernel_values = TARGET_KERNEL_SIZE * TARGET_KERNEL_SIZE
    if optimized:
        depthwise_parameters = STEM_CHANNELS * kernel_values
        pointwise_parameters = STEM_CHANNELS * TARGET_CHANNELS
        pointwise_macs = spatial_positions * pointwise_parameters
        depthwise_macs = spatial_positions * depthwise_parameters
        return depthwise_parameters + pointwise_parameters, depthwise_macs + pointwise_macs
    parameters = TARGET_CHANNELS * STEM_CHANNELS * kernel_values
    macs = spatial_positions * parameters
    return parameters, macs


def export_model(model, model_name):
    """Export one model with named ONNX input and output tensors."""
    model.eval()
    sample = torch.zeros(WARMUP_BATCH, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    output_path = OUTPUT_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        (sample,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))
    return output_path


def main():
    """Export both graph variants and assert their shapes and target MAC totals."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for optimized, model_name in ((False, "original"), (True, "optimized")):
        model = SimpleCNN(optimized=optimized).eval()
        sample = torch.zeros(WARMUP_BATCH, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        with torch.inference_mode():
            scores = model(sample)
        assert tuple(scores.shape) == (WARMUP_BATCH, CLASS_COUNT)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        parameters, macs = target_layer_metrics(optimized)
        expected_macs = (
            EXPECTED_OPTIMIZED_TARGET_MACS
            if optimized
            else EXPECTED_ORIGINAL_TARGET_MACS
        )
        assert macs == expected_macs
        output_path = export_model(model, model_name)
        print(
            f"{model_name}: total parameters={total_parameters:,}, "
            f"target weights={parameters:,}, target MACs={macs:,}, "
            f"model={output_path.stat().st_size / (1024 * 1024):.3f} MiB"
        )


if __name__ == "__main__":
    main()
