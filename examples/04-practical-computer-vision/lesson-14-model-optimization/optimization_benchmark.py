"""Compare model precision and inference latency for a parcel-camera workload."""

from copy import deepcopy
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
MODEL_INPUT_SHORT_EDGE = 224
WARMUP_STEPS = 10
MEASURED_STEPS = 40
SEED = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_parcel_frame():
    """Draw a sample conveyor-camera frame to serve as model input."""
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
    draw.text((18, 18), "PARCEL CAMERA · SAMPLE FRAME", fill=(20, 35, 45))
    return image


def prepare_input(image, device, dtype):
    """Resize and normalize one camera frame for MobileNet inference."""
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
    pixels = pixels.unsqueeze(0).to(device=device, dtype=dtype).div_(255)
    input_height = round(MODEL_INPUT_SHORT_EDGE * CAMERA_HEIGHT / CAMERA_WIDTH)
    pixels = functional.interpolate(
        pixels, size=(input_height, MODEL_INPUT_SHORT_EDGE), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


# --8<-- [start:measure_latency]
def measure_latency(model, inputs, device):
    """Return median synchronized model-only latency in milliseconds."""
    with torch.inference_mode():
        for _ in range(WARMUP_STEPS):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(MEASURED_STEPS):
            started = perf_counter()
            model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((perf_counter() - started) * 1000)
    return median(samples)
# --8<-- [end:measure_latency]


def parameter_storage_estimates(model):
    """Estimate parameter bytes at FP32, FP16, and INT8 precision."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "FP32": parameter_count * 4 / 1_000_000,
        "FP16": parameter_count * 2 / 1_000_000,
        "INT8 (estimate)": parameter_count / 1_000_000,
    }


def save_comparison(frame, model, latency_results, fp16_error, device_name):
    """Save and display the parcel-camera frame and benchmark comparisons."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(frame)
    axes[0].set_title("Parcel camera workload\n640 × 480 frame → 224 × 168 input")
    axes[0].axis("off")

    storage = parameter_storage_estimates(model)
    axes[1].bar(storage.keys(), storage.values(), color=("#2563eb", "#059669", "#f59e0b"))
    axes[1].set_title("Parameter storage estimate")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=18)
    for index, value in enumerate(storage.values()):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    axes[2].bar(
        latency_results.keys(), latency_results.values(),
        color=("#64748b", "#2563eb", "#059669")[: len(latency_results)],
    )
    axes[2].set_title("Measured single-frame latency\n" + device_name)
    axes[2].set_ylabel("Median milliseconds · lower is faster")
    axes[2].tick_params(axis="x", rotation=18)
    for index, value in enumerate(latency_results.values()):
        axes[2].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    difference_label = (
        f"FP16 max logit difference: {fp16_error:.2e}"
        if fp16_error is not None
        else "FP16 comparison requires CUDA"
    )
    figure.suptitle(f"Precision trade-offs · {difference_label}")
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "optimization_tradeoffs.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def main():
    """Benchmark available precisions and visualize their trade-offs."""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    frame = make_parcel_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    baseline = mobilenet_v3_small(weights=None).eval()

    cpu_model = deepcopy(baseline).to(device="cpu", dtype=torch.float32)
    cpu_input = prepare_input(frame, torch.device("cpu"), torch.float32)
    latency_results = {"CPU FP32": measure_latency(cpu_model, cpu_input, torch.device("cpu"))}

    fp16_error = None
    if device.type == "cuda":
        gpu_model = deepcopy(baseline).to(device=device, dtype=torch.float32)
        gpu_input = prepare_input(frame, device, torch.float32)
        with torch.inference_mode():
            fp32_logits = gpu_model(gpu_input)
        latency_results["GPU FP32"] = measure_latency(gpu_model, gpu_input, device)

        half_model = deepcopy(baseline).to(device=device, dtype=torch.float16)
        half_input = prepare_input(frame, device, torch.float16)
        with torch.inference_mode():
            fp16_logits = half_model(half_input).float()
        fp16_error = (fp32_logits - fp16_logits).abs().max().item()
        latency_results["GPU FP16"] = measure_latency(half_model, half_input, device)

    assert all(latency > 0 for latency in latency_results.values())
    storage = parameter_storage_estimates(baseline)
    assert storage["FP32"] > storage["FP16"] > storage["INT8 (estimate)"]
    print("Device:", device_name)
    for mode, latency in latency_results.items():
        print(f"{mode}: {latency:.2f} ms ({1000 / latency:.1f} frames/s)")
    if fp16_error is not None:
        print(f"FP16 maximum absolute logit difference: {fp16_error:.5f}")
    save_comparison(frame, baseline, latency_results, fp16_error, device_name)
    frame.save(OUTPUT_DIR / "parcel_camera_frame.png")
    print("Saved benchmark graph to:", OUTPUT_DIR / "optimization_tradeoffs.png")


if __name__ == "__main__":
    main()
