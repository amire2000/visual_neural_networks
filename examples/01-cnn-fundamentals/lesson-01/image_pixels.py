"""Show how a tiny grayscale image is represented by pixel values."""

image = [
    [0, 0, 255, 255, 255],
    [0, 0, 0, 0, 255],
    [0, 0, 0, 255, 0],
    [0, 0, 255, 0, 0],
    [0, 255, 0, 0, 0],
]

height = len(image)
width = len(image[0])
bright_pixels = sum(pixel > 0 for row in image for pixel in row)

print(f"Image size: {width} x {height}")
print(f"Pixel values: {width * height}")
print(f"Bright pixels: {bright_pixels}")
