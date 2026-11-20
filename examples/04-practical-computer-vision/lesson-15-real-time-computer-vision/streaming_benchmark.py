"""Compare latency and throughput for a conveyor-camera inference stream."""

from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from torchvision.models import mobilenet_v3_small

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
INPUT_RESOLUTIONS = (224, 320)
BATCH_SIZES = (1, 2, 4, 8)
WARMUP_STEPS = 3
MEASURED_STEPS = 20
SEED = 15
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_parcel_frame():
    """Draw a sample conveyor-camera frame for the inference workload."""
    image = Image.new("RGB", (CAMERA_WIDTH, CAMERA_HEIGHT), (205, 215, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 270, CAMERA_WIDTH, CAMERA_HEIGHT), fill=(55, 62, 68))
    draw.rectangle((0, 315, CAMERA_WIDTH, 330), fill=(230, 165, 55))
    for x, color, width, height in (
        (85, (193, 143, 91), 135, 100),
        (275, (174, 187, 194), 150, 130),
        (490, (203, 179, 143), 112, 90),
    ):
        y = 265 - height
        draw.rectangle((x, y, x + width, 265), fill=color, outline=(55, 55, 55), width=3)
        draw.rectangle((x + width // 2 - 5, y, x + width // 2 + 5, 265), fill=(218, 177, 91))
        draw.rectangle((x + 16, y + 18, x + 45, y + 36), fill=(240, 240, 230))
    draw.text((18, 18), "CONVEYOR CAMERA · 30 FPS", fill=(20, 35, 45))
    return image


def frame_tensor(image):
    """Convert one RGB camera frame to a channel-first float tensor."""
    return (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .float()
        .div_(255)
        .unsqueeze(0)
    )


# --8<-- [start:run_pipeline]
def run_pipeline(frame_batch, model, device, resolution):
    """Transfer, resize, normalize, infer, and return predicted class IDs."""
    batch = frame_batch.to(device)
    input_height = round(resolution * CAMERA_HEIGHT / CAMERA_WIDTH)
    batch = functional.interpolate(
        batch, size=(input_height, resolution), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    logits = model((batch - mean) / std)
    return logits.argmax(dim=1).cpu()
# --8<-- [end:run_pipeline]


def measure_pipeline(frame_batch, model, device, resolution):
    """Return median full-pipeline batch latency in milliseconds."""
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            run_pipeline(frame_batch, model, device, resolution)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(MEASURED_STEPS):
            started = perf_counter()
            run_pipeline(frame_batch, model, device, resolution)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((perf_counter() - started) * 1000)
    return median(samples)


def save_comparison(frame, results, device_name):
    """Save and display the camera scene and real-time performance graphs."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(frame)
    axes[0].set_title("Parcel conveyor camera\n640 × 480 RGB frames")
    axes[0].axis("off")

    colors = {224: "#2563eb", 320: "#ea580c"}
    for resolution in INPUT_RESOLUTIONS:
        response_times = [results[(resolution, size)]["response_ms"] for size in BATCH_SIZES]
        axes[1].plot(BATCH_SIZES, response_times, marker="o", label=f"short edge {resolution}px", color=colors[resolution])
        throughputs = [results[(resolution, size)]["fps"] for size in BATCH_SIZES]
        axes[2].plot(BATCH_SIZES, throughputs, marker="o", label=f"short edge {resolution}px", color=colors[resolution])

    frame_interval = 1000 / CAMERA_FPS
    axes[1].axhline(frame_interval, color="#dc2626", linestyle="--", label=f"{CAMERA_FPS} FPS frame interval")
    axes[1].set_title("Estimated frame response time")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Pipeline time + average batch wait (ms)")
    axes[1].set_xticks(BATCH_SIZES)
    axes[1].legend(fontsize=8)

    axes[2].axhline(CAMERA_FPS, color="#dc2626", linestyle="--", label=f"Camera input: {CAMERA_FPS} FPS")
    axes[2].set_title("Measured service throughput")
    axes[2].set_xlabel("Batch size")
    axes[2].set_ylabel("Frames per second")
    axes[2].set_xticks(BATCH_SIZES)
    axes[2].legend(fontsize=8)

    figure.suptitle(f"Batching and resolution trade-offs · {device_name}")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "streaming_tradeoffs.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main():
    """Measure the parcel-camera pipeline for each resolution and batch size."""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    frame = make_parcel_frame()
    raw_frame = frame_tensor(frame)
    model = mobilenet_v3_small(weights=None).eval().to(device)
    results = {}

    for resolution in INPUT_RESOLUTIONS:
        for batch_size in BATCH_SIZES:
            frame_batch = raw_frame.repeat(batch_size, 1, 1, 1)
            batch_latency = measure_pipeline(frame_batch, model, device, resolution)
            average_fill_wait = (batch_size - 1) / (2 * CAMERA_FPS) * 1000
            response_time = batch_latency + average_fill_wait
            throughput = batch_size * 1000 / batch_latency
            results[(resolution, batch_size)] = {
                "batch_ms": batch_latency,
                "wait_ms": average_fill_wait,
                "response_ms": response_time,
                "fps": throughput,
            }
            print(
                f"{resolution}px, batch {batch_size}: "
                f"pipeline={batch_latency:.1f} ms, average fill wait={average_fill_wait:.1f} ms, "
                f"response={response_time:.1f} ms, throughput={throughput:.1f} FPS"
            )

    assert len(results) == len(INPUT_RESOLUTIONS) * len(BATCH_SIZES)
    assert all(result["batch_ms"] > 0 and result["fps"] > 0 for result in results.values())
    save_comparison(frame, results, device_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.save(OUTPUT_DIR / "parcel_camera_frame.png")
    print("Saved streaming graph to:", OUTPUT_DIR / "streaming_tradeoffs.png")


if __name__ == "__main__":
    main()
