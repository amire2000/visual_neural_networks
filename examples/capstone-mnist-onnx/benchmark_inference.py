"""Compare CPU loading and inference times for PyTorch and ONNX model files."""

from time import perf_counter

import onnxruntime as ort
import torch

from infer_onnx import INFERENCE_BATCH_SIZE, make_test_batch
from train_mnist_mlp import MNISTMLP, NUM_CLASSES, OUTPUT_DIR

FLOAT_ONNX_PATH = OUTPUT_DIR / "mnist_mlp.onnx"
INT8_ONNX_PATH = OUTPUT_DIR / "mnist_mlp_int8.onnx"
CHECKPOINT_PATH = OUTPUT_DIR / "best_mlp.pth"
WARMUP_RUNS = 10
TIMED_RUNS = 100
CPU_THREADS = 1
INPUT_NAME = "images"
OUTPUT_NAME = "digit_scores"


def load_onnx_session(model_path):
    """Load an ONNX model with a single-threaded CPU execution provider."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def time_torch_model(model, images):
    """Warm up a PyTorch model and return average batch latency in milliseconds."""
    with torch.inference_mode():
        # Warmup avoids including first-use setup in the timed measurement.
        for _ in range(WARMUP_RUNS):
            logits = model(images)
        assert tuple(logits.shape) == (INFERENCE_BATCH_SIZE, NUM_CLASSES)
        start_time = perf_counter()
        for _ in range(TIMED_RUNS):
            logits = model(images)
        elapsed_seconds = perf_counter() - start_time
    return 1000.0 * elapsed_seconds / TIMED_RUNS, logits.numpy()


def time_onnx_session(session, images):
    """Warm up an ONNX session and return average batch latency in milliseconds."""
    # Prepare the same CPU NumPy input once, outside the timed inference loop.
    input_array = images.numpy()
    for _ in range(WARMUP_RUNS):
        scores = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
    assert scores.shape == (INFERENCE_BATCH_SIZE, NUM_CLASSES)
    start_time = perf_counter()
    for _ in range(TIMED_RUNS):
        scores = session.run([OUTPUT_NAME], {INPUT_NAME: input_array})[0]
    elapsed_seconds = perf_counter() - start_time
    return 1000.0 * elapsed_seconds / TIMED_RUNS, scores


def format_result(name, artifact, runtime, load_ms, batch_ms, scores, labels):
    """Format model timings, throughput, and accuracy as a Markdown table row."""
    latency_ms = batch_ms / INFERENCE_BATCH_SIZE
    throughput = INFERENCE_BATCH_SIZE * 1000.0 / batch_ms
    accuracy = 100.0 * (scores.argmax(axis=1) == labels.numpy()).mean()
    return (
        f"| {name} | `{artifact}` | {runtime} | {load_ms:.2f} | "
        f"{batch_ms:.3f} | {latency_ms:.3f} | {throughput:.1f} | {accuracy:.2f}% |"
    )


def main():
    """Measure all three models on the same MNIST test batch and print a table."""
    # Require the trained checkpoint and both exported artifacts before benchmarking.
    model_paths = (CHECKPOINT_PATH, FLOAT_ONNX_PATH, INT8_ONNX_PATH)
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path}. Train, export, and quantize the model first."
            )

    # Use one CPU thread in both runtimes for a more comparable small-model run.
    torch.set_num_threads(CPU_THREADS)
    images, labels = make_test_batch()
    rows = []

    # Time model construction and checkpoint loading separately from inference.
    start_time = perf_counter()
    torch_model = MNISTMLP().cpu().eval()
    torch_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    torch_load_ms = 1000.0 * (perf_counter() - start_time)
    torch_batch_ms, torch_scores = time_torch_model(torch_model, images)
    rows.append(
        format_result(
            "PyTorch FP32", CHECKPOINT_PATH.name, "PyTorch CPU", torch_load_ms,
            torch_batch_ms, torch_scores, labels,
        )
    )

    for model_path, model_name in (
        (FLOAT_ONNX_PATH, "ONNX FP32"),
        (INT8_ONNX_PATH, "ONNX int8"),
    ):
        # Session creation includes reading the graph and runtime initialization.
        start_time = perf_counter()
        session = load_onnx_session(model_path)
        load_ms = 1000.0 * (perf_counter() - start_time)
        batch_ms, scores = time_onnx_session(session, images)
        rows.append(
            format_result(
                model_name, model_path.name, "ONNX Runtime CPU", load_ms,
                batch_ms, scores, labels,
            )
        )

    print(f"CPU benchmark, batch size {INFERENCE_BATCH_SIZE}, {TIMED_RUNS} timed runs")
    print("Times exclude MNIST loading and preprocessing. ONNX quantization time is excluded.")
    print("| Model | Loaded artifact | Runtime | Load ms | Batch ms | ms/image | Images/s | Batch accuracy |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
