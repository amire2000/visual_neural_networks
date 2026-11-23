"""CIFAR-10 ResNet-20 and a small CNN comparison model."""

import torch
from torch import nn

IMAGE_SIZE = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10
BLOCKS_PER_STAGE = 3
STAGE_CHANNELS = (16, 32, 64)


class ResidualBlock(nn.Module):
    """Learn a change to the input and add it to a shape-matched shortcut."""

    def __init__(self, input_channels, output_channels, stride=1):
        """Create two convolution steps and adjust the shortcut when needed."""
        super().__init__()
        # --8<-- [start:block]
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        # --8<-- [end:block]

    def forward(self, images):
        """Return the learned path plus the identity or projected shortcut."""
        # --8<-- [start:block_forward]
        change = self.relu(self.bn1(self.conv1(images)))
        change = self.bn2(self.conv2(change))
        return self.relu(change + self.shortcut(images))
        # --8<-- [end:block_forward]


class ResNet20(nn.Module):
    """Classify CIFAR-10 images with the paper's 6n+2 small-image design."""

    def __init__(self):
        """Build three stages with three two-convolution blocks per stage."""
        super().__init__()
        # --8<-- [start:network]
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, STAGE_CHANNELS[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(STAGE_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        input_channels = STAGE_CHANNELS[0]
        for stage_index, output_channels in enumerate(STAGE_CHANNELS):
            for block_index in range(BLOCKS_PER_STAGE):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(ResidualBlock(input_channels, output_channels, stride))
                input_channels = output_channels
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(STAGE_CHANNELS[-1], NUM_CLASSES)
        # --8<-- [end:network]

    def forward(self, images):
        """Return ten class scores for every image in the batch."""
        # --8<-- [start:network_forward]
        features = self.stages(self.stem(images))
        return self.classifier(torch.flatten(self.pool(features), 1))
        # --8<-- [end:network_forward]


class SmallCNN(nn.Module):
    """Provide a plain convolutional baseline for the residual model."""

    def __init__(self):
        """Create three convolution blocks and a global-pooling classifier."""
        super().__init__()
        layers = []
        input_channels = INPUT_CHANNELS
        for output_channels in STAGE_CHANNELS:
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(STAGE_CHANNELS[-1], NUM_CLASSES)

    def forward(self, images):
        """Return ten class scores for every image in the batch."""
        features = self.features(images)
        return self.classifier(torch.flatten(self.pool(features), 1))


def check_model_shapes():
    """Assert model outputs and residual shortcuts have compatible shapes."""
    images = torch.zeros(2, INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    for model in (ResNet20(), SmallCNN()):
        assert tuple(model(images).shape) == (2, NUM_CLASSES)
    downsample = ResidualBlock(16, 32, stride=2)
    assert tuple(downsample(images.new_zeros(2, 16, 32, 32)).shape) == (2, 32, 16, 16)


if __name__ == "__main__":
    check_model_shapes()
    print("ResNet-20 and SmallCNN: (2, 3, 32, 32) → (2, 10)")
