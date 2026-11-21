# --8<-- [start:model]
"""A compact Vision Transformer for 28 by 28 MNIST images."""

import torch
from torch import nn

IMAGE_SIZE = 28
INPUT_CHANNELS = 1
PATCH_SIZE = 7
PATCHES_PER_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = PATCHES_PER_SIDE * PATCHES_PER_SIDE
PATCH_FEATURES = PATCH_SIZE * PATCH_SIZE
EMBED_DIM = 64
ATTENTION_HEADS = 4
FEEDFORWARD_DIM = 128
ENCODER_LAYERS = 2
NUM_CLASSES = 10


class MNISTVisionTransformer(nn.Module):
    """Split MNIST into patches and classify the encoded class token."""

    def __init__(self):
        """Create patch, position, encoder, and digit-classification layers."""
        super().__init__()
        # --8<-- [start:model]
        self.patchify = nn.Unfold(kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.patch_projection = nn.Linear(PATCH_FEATURES, EMBED_DIM)
        self.class_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.position_embedding = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, EMBED_DIM))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=ENCODER_LAYERS)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
        # Give each cloned encoder layer its own initial projection weights.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        # --8<-- [end:model]

    def forward(self, images):
        """Return ten digit logits for each image in a batch."""
        # Each 28×28 image becomes 16 flattened 7×7 grayscale patches.
        patches = self.patchify(images).transpose(1, 2)
        tokens = self.patch_projection(patches)
        class_tokens = self.class_token.expand(images.size(0), -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded_tokens = self.encoder(tokens)
        return self.classifier(encoded_tokens[:, 0])
# --8<-- [end:model]
