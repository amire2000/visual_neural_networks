"""Export baseline and narrower-channel CNNs for the Netron channel-width lab."""

from pathlib import Path

import onnx
import torch
from torch import nn

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
INPUT_CHANNELS = 3
STEM_CHANNELS = 32
BASELINE_WIDTH = 64
NARROW_WIDTH = 32
CLASS_COUNT = 10
INPUT_SIZE = 224
STEM_KERNEL_SIZE = 3
STEM_STRIDE = 2
STEM_PADDING = 1
FEATURE_KERNEL_SIZE = 3
FEATURE_PADDING = 1
POOL_OUTPUT_SIZE = 1
ONNX_OPSET = 17
BATCH_SIZE = 1
EXPECTED_PARAMETERS = {"baseline": 20_042, "narrow": 10_474}
EXPECTED_TARGET_MACS = {"baseline": 231_211_008, "narrow": 115_605_504}
TARGET_SPATIAL_SIZE = INPUT_SIZE // STEM_STRIDE


# --8<-- [start:model]
class ChannelCNN(nn.Module):
    """Classify one RGB image with a selectable number of feature channels."""

    def __init__(self, feature_channels=BASELINE_WIDTH):
        """Build the CNN and match the classifier input to the feature width."""
        super().__init__()
        if feature_channels not in (BASELINE_WIDTH, NARROW_WIDTH):
            raise ValueError(f"Unsupported feature width: {feature_channels}")
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
        self.feature = nn.Conv2d(
            STEM_CHANNELS,
            feature_channels,
            FEATURE_KERNEL_SIZE,
            padding=FEATURE_PADDING,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(POOL_OUTPUT_SIZE)
        # Narrowing the feature map also narrows the classifier's input vector.
        self.classifier = nn.Linear(feature_channels, CLASS_COUNT)

    def forward(self, images):
        """Return ten class scores for every image in the batch."""
        features = self.stem(images)
        features = self.relu(self.feature(features))
        features = self.pool(features)
        return self.classifier(torch.flatten(features, 1))
# --8<-- [end:model]


def target_layer_metrics(feature_channels):
    """Calculate target-convolution weights, parameters, and MACs."""
    kernel_values = FEATURE_KERNEL_SIZE * FEATURE_KERNEL_SIZE
    weights = feature_channels * STEM_CHANNELS * kernel_values
    parameters = weights + feature_channels
    macs = TARGET_SPATIAL_SIZE**2 * weights
    return weights, parameters, macs


def export_model(model, model_name):
    """Export one fixed-shape model and validate its ONNX graph."""
    model.eval()
    sample = torch.zeros(BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        scores = model(sample)
    assert tuple(scores.shape) == (BATCH_SIZE, CLASS_COUNT)
    output_path = OUTPUT_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        (sample,),
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))
    return output_path


def main():
    """Export both channel widths and check their expected size and compute."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = (("baseline", BASELINE_WIDTH), ("narrow", NARROW_WIDTH))
    print("| Model | Feature width | Total parameters | Target MACs | ONNX MiB |")
    print("|---|---:|---:|---:|---:|")
    for model_name, width in variants:
        model = ChannelCNN(width).eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        weights, target_parameters, macs = target_layer_metrics(width)
        assert parameter_count == EXPECTED_PARAMETERS[model_name]
        assert macs == EXPECTED_TARGET_MACS[model_name]
        assert target_parameters == weights + width
        output_path = export_model(model, model_name)
        print(
            f"| {model_name} | {width} | {parameter_count:,} | {macs:,} | "
            f"{output_path.stat().st_size / (1024 * 1024):.3f} |"
        )


if __name__ == "__main__":
    main()
