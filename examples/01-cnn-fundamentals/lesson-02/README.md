# Lesson 2: Filter response example

## Prerequisites

Python 3.11 or later and the course environment created with `uv sync`. The example uses only the Python standard library.

## Run

From the repository root:

```bash
uv run python examples/01-cnn-fundamentals/lesson-02/filter_response.py
```

## Expected output

```text
Random filter response: -0.1
Vertical-line filter response: 6
Uniform-bright patch response: 0
```

## Limitations

This script calculates one 3 × 3 filter response. It does not implement convolution over a full image or train a CNN.
