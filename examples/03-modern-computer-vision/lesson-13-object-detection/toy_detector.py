"""Draw a simple scene and localize colored toy objects with bounding boxes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SCENE_WIDTH = 640
SCENE_HEIGHT = 360
OBJECT_COLORS = {
    "car": (225, 55, 55),
    "ball": (45, 110, 225),
    "tree": (40, 175, 85),
}
BOX_COLORS = {"car": "#dc2626", "ball": "#2563eb", "tree": "#16a34a"}


def make_scene():
    """Create a colorful toy scene with three objects in distinct colors."""
    image = Image.new("RGB", (SCENE_WIDTH, SCENE_HEIGHT), (185, 220, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 255, SCENE_WIDTH, SCENE_HEIGHT), fill=(220, 230, 205))
    draw.polygon(
        ((70, 285), (70, 235), (112, 235), (138, 210), (210, 210), (240, 240), (250, 285)),
        fill=OBJECT_COLORS["car"],
    )
    draw.ellipse((380, 105, 505, 230), fill=OBJECT_COLORS["ball"])
    draw.ellipse((520, 100, 620, 205), fill=OBJECT_COLORS["tree"])
    return image


# --8<-- [start:find_boxes]
def find_color_boxes(image):
    """Return class names and tight boxes around pixels of each target color."""
    pixels = np.asarray(image)
    detections = []
    for class_name, color in OBJECT_COLORS.items():
        rows, columns = np.where(np.all(pixels == color, axis=2))
        if len(rows):
            x_min, x_max = int(columns.min()), int(columns.max())
            y_min, y_max = int(rows.min()), int(rows.max())
            detections.append((class_name, x_min, y_min, x_max + 1, y_max + 1))
    return detections
# --8<-- [end:find_boxes]


def save_detection_figure(image, detections):
    """Save and display the original scene and its labeled boxes."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Input scene")
    axes[1].imshow(image)
    axes[1].set_title("Color regions localized as objects")
    for class_name, x_min, y_min, x_max, y_max in detections:
        box = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=BOX_COLORS[class_name],
            linewidth=3,
        )
        axes[1].add_patch(box)
        axes[1].text(
            x_min,
            max(14, y_min - 5),
            class_name,
            color="white",
            backgroundcolor=BOX_COLORS[class_name],
            fontsize=10,
        )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "toy_detections.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main():
    """Detect the colored toy objects and display their bounding boxes."""
    image = make_scene()
    detections = find_color_boxes(image)
    for class_name, x_min, y_min, x_max, y_max in detections:
        print(f"{class_name}: box=({x_min}, {y_min}, {x_max}, {y_max})")
    save_detection_figure(image, detections)
    print("Saved and displayed:", OUTPUT_DIR / "toy_detections.png")


if __name__ == "__main__":
    main()
