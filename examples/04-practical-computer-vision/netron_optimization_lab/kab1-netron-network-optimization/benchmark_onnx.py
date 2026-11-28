"""Compare CPU ONNX Runtime startup and latency for both exported graphs."""

from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import onnxruntime as ort

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAMES = ("original", "optimized")
INPUT_CHANNELS = 3
INPUT_SIZE = 224
BATCH_SIZE = 1
WARMUP_RUNS = 20
TIMED_RUNS = 100
CPU_THREADS = 1
INPUT_NAME = "input"
OUTPUT_NAME = "output"
SEED = 23


def benchmark(model_name, images):
    """Return session startup time, median inference latency, and file size."""
    model_path = OUTPUT_DIR / f"{model_name}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run export_models.py before benchmarking."
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    started = perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    load_ms = (perf_counter() - started) * 1000
    feed = {INPUT_NAME: images}
    for _ in range(WARMUP_RUNS):
        session.run([OUTPUT_NAME], feed)
    samples = []
    for _ in range(TIMED_RUNS):
        started = perf_counter()
        session.run([OUTPUT_NAME], feed)
        samples.append((perf_counter() - started) * 1000)
    return load_ms, median(samples), model_path.stat().st_size / (1024 * 1024)


def main():
    """Benchmark both ONNX graphs on the same seeded input tensor."""
    images = np.random.default_rng(SEED).standard_normal(
        (BATCH_SIZE, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE), dtype=np.float32
    )
    print("| Model | Session load (ms) | Median inference (ms) | Model (MiB) |")
    print("|---|---:|---:|---:|")
    results = {}
    for model_name in MODEL_NAMES:
        load_ms, latency_ms, size_mb = benchmark(model_name, images)
        results[model_name] = latency_ms
        print(f"| {model_name} | {load_ms:.2f} | {latency_ms:.3f} | {size_mb:.3f} |")
    assert all(latency > 0 for latency in results.values())
    print(f"Median latency ratio (original / optimized): {results['original'] / results['optimized']:.2f}×")


if __name__ == "__main__":
    main()
