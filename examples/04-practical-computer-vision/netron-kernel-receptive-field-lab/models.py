"""Export a baseline 5×5 CNN and a two-layer 3×3 variant."""

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
VARIANTS = ("baseline", "optimized")
INPUT_CHANNELS = 3
STEM_CHANNELS = 32
FEATURE_CHANNELS = 64
CLASS_COUNT = 10
INPUT_SIZE = 224
STEM_KERNEL = 3
STEM_STRIDE = 2
STEM_PADDING = 1
LARGE_KERNEL = 5
LARGE_PADDING = 2
SMALL_KERNEL = 3
SMALL_PADDING = 1
POOL_SIZE = 1
BATCH_SIZE = 1
ONNX_OPSET = 17
EXPECTED_PARAMETERS = {"baseline": 52_810, "optimized": 29_290}
EXPECTED_TARGET_MACS = {"baseline": 642_252_800, "optimized": 346_816_512}
TARGET_SPATIAL_SIZE = INPUT_SIZE // STEM_STRIDE


# --8<-- [start:model]
class KernelCNN(nn.Module):
    """Classify an RGB image with one 5×5 or two stacked 3×3 convolutions."""

    def __init__(self, variant="baseline"):
        """Build the requested target block and keep the classifier interface fixed."""
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.stem = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                STEM_CHANNELS,
                STEM_KERNEL,
                stride=STEM_STRIDE,
                padding=STEM_PADDING,
            ),
            nn.ReLU(),
        )
        if variant == "baseline":
            self.feature = nn.Sequential(
                nn.Conv2d(
                    STEM_CHANNELS,
                    FEATURE_CHANNELS,
                    LARGE_KERNEL,
                    padding=LARGE_PADDING,
                ),
                nn.ReLU(),
            )
        else:
            # Two 3×3 layers reach a 5×5 theoretical receptive field, with an extra ReLU.
            self.feature = nn.Sequential(
                nn.Conv2d(
                    STEM_CHANNELS,
                    STEM_CHANNELS,
                    SMALL_KERNEL,
                    padding=SMALL_PADDING,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    STEM_CHANNELS,
                    FEATURE_CHANNELS,
                    SMALL_KERNEL,
                    padding=SMALL_PADDING,
                ),
                nn.ReLU(),
            )
        self.pool = nn.AdaptiveAvgPool2d(POOL_SIZE)
        self.classifier = nn.Linear(FEATURE_CHANNELS, CLASS_COUNT)

    def forward(self, images):
        """Return ten class scores while preserving the external output shape."""
        features = self.stem(images)
        features = self.feature(features)
        features = self.pool(features)
        return self.classifier(torch.flatten(features, 1))
# --8<-- [end:model]


def target_layer_macs(variant):
    """Calculate target-block MACs for the selected convolution sequence."""
    spatial_positions = TARGET_SPATIAL_SIZE**2
    one_conv_3x3_macs = (
        spatial_positions * STEM_CHANNELS * FEATURE_CHANNELS * SMALL_KERNEL**2
    )
    if variant == "baseline":
        return spatial_positions * FEATURE_CHANNELS * STEM_CHANNELS * LARGE_KERNEL**2
    return (
        spatial_positions * STEM_CHANNELS * STEM_CHANNELS * SMALL_KERNEL**2
        + one_conv_3x3_macs
    )


def export_model(variant):
    """Export one CNN variant and validate the ONNX graph and output shape."""
    model = KernelCNN(variant).eval()
    sample = torch.zeros(BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        logits = model(sample)
    assert tuple(logits.shape) == (BATCH_SIZE, CLASS_COUNT)
    model_path = OUTPUT_DIR / f"{variant}.onnx"
    torch.onnx.export(
        model,
        (sample,),
        str(model_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(model_path)))
    return model, model_path


def main():
    """Export both variants and check expected parameter and MAC totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = VARIANTS if arguments.variant == "both" else (arguments.variant,)
    print("| Model | Total parameters | Target MACs | ONNX MiB |")
    print("|---|---:|---:|---:|")
    for variant in variants:
        model, model_path = export_model(variant)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        macs = target_layer_macs(variant)
        assert parameters == EXPECTED_PARAMETERS[variant]
        assert macs == EXPECTED_TARGET_MACS[variant]
        print(
            f"| {variant} | {parameters:,} | {macs:,} | "
            f"{model_path.stat().st_size / (1024 * 1024):.3f} |"
        )


if __name__ == "__main__":
    main()
