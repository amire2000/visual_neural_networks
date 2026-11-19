# Lesson 6: Inspect training history

## Prerequisites

Python 3.11 or later and a completed Lesson 4 training run. The example uses only the Python standard library and reads `examples/01-cnn-fundamentals/lesson-04/outputs/training_history.csv`.

## Run

From the repository root:

```bash
uv run python examples/01-cnn-fundamentals/lesson-06/analyze_history.py
```

## Expected output

The script reports the epoch with the lowest validation loss, the final training and validation accuracy, and their gap. The exact values depend on the training run.

## Limitations

The summary is descriptive, not an automatic overfitting diagnosis. Read it together with `training_curves.png` and keep test metrics out of tuning decisions.
