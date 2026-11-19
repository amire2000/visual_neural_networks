"""Trace tensor shapes and parameter counts for the lesson's MNIST CNN."""


def conv2d_shape(shape, out_channels, kernel_size=3, padding=1, stride=1):
    """Return the output channels, height, and width for a 2D convolution."""
    channels, height, width = shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    return out_channels, out_height, out_width


def max_pool2d_shape(shape, kernel_size=2, stride=2):
    """Return the output shape after a 2D max-pooling operation."""
    channels, height, width = shape
    return channels, (height - kernel_size) // stride + 1, (width - kernel_size) // stride + 1


def conv_parameters(in_channels, out_channels, kernel_size):
    """Count convolution weights and one bias per output channel."""
    return out_channels * (in_channels * kernel_size * kernel_size + 1)


shape = (1, 28, 28)
parameter_counts = []
print(f"Input: {shape}")

shape = conv2d_shape(shape, out_channels=16)
parameter_counts.append(conv_parameters(1, 16, 3))
print(f"Conv2D + ReLU: {shape}")

shape = max_pool2d_shape(shape)
print(f"MaxPool2D: {shape}")

shape = conv2d_shape(shape, out_channels=32)
parameter_counts.append(conv_parameters(16, 32, 3))
print(f"Conv2D + ReLU: {shape}")

shape = max_pool2d_shape(shape)
print(f"MaxPool2D: {shape}")

flattened_size = shape[0] * shape[1] * shape[2]
print(f"Flatten: ({flattened_size},)")
linear_parameters = flattened_size * 10 + 10
parameter_counts.append(linear_parameters)
print(f"Linear: (10,)")
print(f"Final output: (10,)")
print(f"Total learnable parameters: {sum(parameter_counts)}")

assert shape == (32, 7, 7)
assert flattened_size == 1568
assert sum(parameter_counts) == 20490
