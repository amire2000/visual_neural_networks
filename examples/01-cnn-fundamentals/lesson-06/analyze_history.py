"""Summarize the training history produced by the Lesson 4 MNIST script."""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = PROJECT_ROOT / "examples/01-cnn-fundamentals/lesson-04/outputs/training_history.csv"


def read_history(path):
    """Read per-epoch metrics from the Lesson 4 CSV file."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    """Print the best validation epoch and the final accuracy gap."""
    if not HISTORY_PATH.exists():
        raise SystemExit("Run the Lesson 4 training script first to create training_history.csv.")

    history = read_history(HISTORY_PATH)
    assert history, "Training history must contain at least one epoch"
    best_epoch = min(history, key=lambda row: float(row["validation_loss"]))
    final = history[-1]
    accuracy_gap = float(final["train_accuracy"]) - float(final["validation_accuracy"])

    print(
        "Lowest validation loss: "
        f"epoch {best_epoch['epoch']} ({float(best_epoch['validation_loss']):.4f})"
    )
    print(f"Final training accuracy: {float(final['train_accuracy']):.2f}%")
    print(f"Final validation accuracy: {float(final['validation_accuracy']):.2f}%")
    print(f"Final train-validation accuracy gap: {accuracy_gap:.2f} percentage points")
    print(f"Learning rate: {float(final['learning_rate']):g}")
    print("Use the curves to decide whether the gap or validation trend is concerning.")


if __name__ == "__main__":
    main()
