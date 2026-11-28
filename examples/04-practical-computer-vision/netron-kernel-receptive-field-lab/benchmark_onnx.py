"""Compare ONNX Runtime startup and CPU inference for two kernel designs."""

from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import onnxruntime as ort

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_NAMES = ("baseline", "optimized")
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input"
OUTPUT_NAME = "logits"
CPU_THREADS = 1
WARMUP_RUNS = 20
TIMED_RUNS = 100
SEED = 42


def benchmark(model_name, images):
    """Return session creation time, median latency, and ONNX file size."""
    model_path = OUTPUT_DIR / f"{model_name}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Export both models first; missing {model_path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = CPU_THREADS
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
        scores = session.run([OUTPUT_NAME], feed)[0]
        assert scores.shape == (INPUT_SHAPE[0], 10)
        samples.append((perf_counter() - started) * 1000)
    return load_ms, median(samples), model_path.stat().st_size / (1024 * 1024)


def main():
    """Benchmark both variants on one reproducible input tensor."""
    images = np.random.default_rng(SEED).standard_normal(INPUT_SHAPE).astype(np.float32)
    results = {}
    print("| Model | Session load (ms) | Median inference (ms) | ONNX MiB |")
    print("|---|---:|---:|---:|")
    for model_name in MODEL_NAMES:
        load_ms, latency_ms, size_mb = benchmark(model_name, images)
        results[model_name] = latency_ms
        print(f"| {model_name} | {load_ms:.2f} | {latency_ms:.3f} | {size_mb:.3f} |")
    assert all(latency > 0 for latency in results.values())
    print(f"Speedup (baseline / optimized): {results['baseline'] / results['optimized']:.2f}×")


if __name__ == "__main__":
    main()
