# Lesson 3: CNN shape trace

## Prerequisites

Python 3.11 or later and the course environment created with `uv sync`. The example uses only the Python standard library.

## Run

From the repository root:

```bash
uv run python examples/01-cnn-fundamentals/lesson-03/shape_trace.py
```

## Expected output

The trace ends with:

```text
Final output: (10,)
Total learnable parameters: 20490
```

## Limitations

This script calculates tensor shapes and parameter counts. It does not create tensors, implement layers, or train a model.
