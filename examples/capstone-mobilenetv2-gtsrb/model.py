"""Small MobileNetV2 and CNN baseline for traffic-sign classification."""

import torch
from torch import nn

IMAGE_SIZE = 96
INPUT_CHANNELS = 3
CLASS_COUNT = 43
WIDTH_MULTIPLIER = 0.5
STEM_CHANNELS = 32
LAST_CHANNELS = 1280
# Each tuple is (expansion t, paper output channels c, repeats n, first stride s).
# Only the first block in a stage uses s; the rest keep the same spatial size.
# --8<-- [start:stage_settings]
STAGE_SETTINGS = (
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
)
# --8<-- [end:stage_settings]
BASELINE_CHANNELS = (32, 64, 128)


# --8<-- [start:channel_rounding]
def make_divisible(value, divisor=8):
    """Round channel counts to a multiple of eight without shrinking much."""
    # Hardware libraries commonly have efficient kernels for aligned channel counts.
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return rounded + divisor if rounded < 0.9 * value else rounded
# --8<-- [end:channel_rounding]


# --8<-- [start:inverted_residual]
class InvertedResidual(nn.Module):
    """Expand, depthwise-filter, and linearly project one MobileNetV2 block."""

    def __init__(self, input_channels, output_channels, stride, expansion):
        """Create the block and use a skip connection when shape and stride match."""
        super().__init__()
        # The expansion ratio changes a narrow input into a richer hidden tensor.
        hidden_channels = input_channels * expansion
        layers = []
        if expansion != 1:
            layers.extend(
                (
                    nn.Conv2d(input_channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                )
            )
        layers.extend(
            (
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    3,
                    stride=stride,
                    padding=1,
                    # One spatial filter per channel makes this depthwise convolution.
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                # Linear projection: no ReLU after the narrow bottleneck.
                nn.Conv2d(hidden_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )
        self.layers = nn.Sequential(*layers)
        # Addition requires matching channel and spatial dimensions.
        self.use_residual = stride == 1 and input_channels == output_channels

    def forward(self, images):
        """Apply the transformation and add the input when the shapes align."""
        output = self.layers(images)
        return images + output if self.use_residual else output
# --8<-- [end:inverted_residual]


# --8<-- [start:mobilenetv2]
class MobileNetV2Small(nn.Module):
    """Classify 96-pixel traffic-sign crops with a half-width MobileNetV2."""

    def __init__(self, class_count=CLASS_COUNT):
        """Build the paper's inverted-residual stages and a task-specific head."""
        super().__init__()
        # The stem downsamples once before the stage schedule starts.
        input_channels = make_divisible(STEM_CHANNELS * WIDTH_MULTIPLIER)
        features = [
            nn.Conv2d(INPUT_CHANNELS, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True),
        ]
        for expansion, base_channels, repeats, first_stride in STAGE_SETTINGS:
            # Width scaling keeps the paper's stage pattern while shrinking the model.
            output_channels = make_divisible(base_channels * WIDTH_MULTIPLIER)
            for block_index in range(repeats):
                # Downsample at the first block only; later repeats preserve resolution.
                stride = first_stride if block_index == 0 else 1
                features.append(
                    InvertedResidual(input_channels, output_channels, stride, expansion)
                )
                input_channels = output_channels
        features.extend(
            (
                # Keep the paper's 1280-channel final feature head at width 0.5.
                nn.Conv2d(input_channels, LAST_CHANNELS, 1, bias=False),
                nn.BatchNorm2d(LAST_CHANNELS),
                nn.ReLU6(inplace=True),
            )
        )
        self.features = nn.Sequential(*features)
        # Pooling removes dependence on the final feature-map width and height.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Replace the ImageNet 1000-class output with this task's 43 sign classes.
        self.classifier = nn.Linear(LAST_CHANNELS, class_count)

    def forward(self, images):
        """Return one raw class score for each image in the batch."""
        features = self.features(images)
        return self.classifier(torch.flatten(self.pool(features), 1))
# --8<-- [end:mobilenetv2]


class SmallCNN(nn.Module):
    """Provide a compact baseline CNN with the same input and output contract."""

    def __init__(self, class_count=CLASS_COUNT):
        """Create three convolution blocks followed by global pooling."""
        super().__init__()
        layers = []
        input_channels = INPUT_CHANNELS
        for output_channels in BASELINE_CHANNELS:
            layers.extend(
                (
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(BASELINE_CHANNELS[-1], class_count)

    def forward(self, images):
        """Return one raw class score for each image in the batch."""
        features = self.features(images)
        return self.classifier(torch.flatten(self.pool(features), 1))


def make_model(model_name):
    """Build the requested architecture from its stable command-line name."""
    if model_name == "mobilenetv2":
        return MobileNetV2Small()
    if model_name == "baseline":
        return SmallCNN()
    raise ValueError(f"Unknown model: {model_name}")


def self_check():
    """Assert both architectures map a sample image batch to 43 class scores."""
    batch_size = 2
    sample = torch.zeros(batch_size, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    for model_name in ("baseline", "mobilenetv2"):
        with torch.inference_mode():
            scores = make_model(model_name).eval()(sample)
        assert tuple(scores.shape) == (batch_size, CLASS_COUNT)
        print(f"{model_name}: {tuple(sample.shape)} → {tuple(scores.shape)}")


if __name__ == "__main__":
    self_check()
