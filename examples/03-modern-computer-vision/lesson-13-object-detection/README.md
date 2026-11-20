# Lesson 13: Object Detection

## Prerequisites

Use the course `.venv` with Matplotlib, NumPy, and Pillow installed. The demo creates its own simple scene and needs no downloaded data or pretrained model.

## Run

From the repository root:

```bash
uv run python examples/03-modern-computer-vision/lesson-13-object-detection/toy_detector.py
```

## Expected outputs

The script prints one class and bounding-box coordinate for each detected color region. It saves and displays `outputs/toy_detections.png`, which compares the scene with the localized objects.

## Limitations

This is a classical color-mask demonstration, not a neural network or YOLO. It only recognizes the exact colors used to draw the toy objects and does not handle natural-image variation, occlusion, or confidence scores.
