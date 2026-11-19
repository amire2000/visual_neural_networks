"""Calculate a 3 by 3 filter's response to small image patches."""

import math


def response(region, weights):
    """Return the sum of element-wise products for equally sized matrices."""
    if len(region) != len(weights) or any(
        len(image_row) != len(weight_row)
        for image_row, weight_row in zip(region, weights)
    ):
        raise ValueError("The image region and filter must have the same shape")
    return sum(
        pixel * weight
        for image_row, weight_row in zip(region, weights)
        for pixel, weight in zip(image_row, weight_row)
    )


vertical_line = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
uniform_patch = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
random_filter = [[0.2, -0.3, 0.1], [-0.1, 0.4, 0.2], [0.3, -0.2, 0.1]]
vertical_filter = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]

random_response = response(vertical_line, random_filter)
line_response = response(vertical_line, vertical_filter)
uniform_response = response(uniform_patch, vertical_filter)

assert math.isclose(random_response, -0.1)
assert line_response == 6
assert uniform_response == 0

print(f"Random filter response: {random_response:.1f}")
print(f"Vertical-line filter response: {line_response}")
print(f"Uniform-bright patch response: {uniform_response}")
