# MobileNetV2 Paper-to-GTSRB Capstone

Implement a reduced-width MobileNetV2 from the paper, train it on German traffic-sign crops, compare it with a small CNN, then export and benchmark its ONNX forms.

## Prerequisites

Complete [Setup Lab](../../docs/lesson-00-setup-lab/index.md) and use the repository `.venv` with PyTorch, TorchVision, and Matplotlib. Add the portable CPU ONNX packages from the repository root:

```bash
uv pip install --python .venv/bin/python onnx onnxscript onnxruntime onnxconverter-common
```

The examples use GTSRB downloaded by TorchVision into `data/gtsrb/`. CUDA training is used when the installed PyTorch build supports it; otherwise training and CPU ONNX inference run on CPU.

## Step-by-step run

From the repository root:

```bash
uv run python examples/capstone-mobilenetv2-gtsrb/model.py
uv run python examples/capstone-mobilenetv2-gtsrb/train.py --model baseline
uv run python examples/capstone-mobilenetv2-gtsrb/train.py --model mobilenetv2
uv run python examples/capstone-mobilenetv2-gtsrb/evaluate.py
uv run python examples/capstone-mobilenetv2-gtsrb/export_onnx.py
uv run python examples/capstone-mobilenetv2-gtsrb/optimize_onnx.py
uv run python examples/capstone-mobilenetv2-gtsrb/benchmark_onnx.py
```

To shorten the first exploration run, append `--epochs 1` to each training command. Use the same epoch count for both architectures when comparing them.

## Expected outputs

The ignored `outputs/` folder contains:

- `baseline_best.pth`, `mobilenetv2_best.pth` — checkpoints selected by validation accuracy.
- `baseline_training.png`, `mobilenetv2_training.png` — training and validation curves.
- `comparison.json`, `model_comparison.png`, `test_predictions.png` — official-test model comparison and visual examples.
- `mobilenetv2_fp32.onnx` — exported dynamic-batch graph, checked against PyTorch logits.
- `mobilenetv2_fp16.onnx`, `mobilenetv2_int8.onnx` — converted FP16 and training-calibrated static INT8 Q/DQ graphs.

The benchmark prints load time, median one-batch inference time, per-image latency, throughput, official-test accuracy, and model file size. Its latency uses already-preprocessed input arrays and includes ONNX Runtime's transfers to/from the selected provider. `evaluate.py` reports PyTorch model latency separately. These measurements vary by device and runtime and should be compared only with the recorded conditions.

## Optional CUDA and TensorRT

CUDA ONNX Runtime requires a compatible GPU package and CUDA/cuDNN runtime. Check [the official provider matrix](https://onnxruntime.ai/docs/install/) for the machine; install either `onnxruntime` or `onnxruntime-gpu`, not both distributions into the same environment. The script skips absent providers. CPU FP16 is not supported by ONNX Runtime; CUDA INT8 is attempted only if CUDA is installed and the quantized graph can execute.

For TensorRT 11 on an NVIDIA GPU, install TensorRT using the versions supported by the local system. The FP16 and INT8 ONNX graphs already contain their reduced-precision operations and quantization scales. The following builds and benchmarks engines from those graphs:

```bash
trtexec --onnx=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_fp32.onnx --minShapes=images:1x3x96x96 --optShapes=images:32x3x96x96 --maxShapes=images:64x3x96x96 --saveEngine=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_fp32.plan
trtexec --onnx=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_fp16.onnx --minShapes=images:1x3x96x96 --optShapes=images:32x3x96x96 --maxShapes=images:64x3x96x96 --saveEngine=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_fp16.plan
trtexec --onnx=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_int8.onnx --minShapes=images:1x3x96x96 --optShapes=images:32x3x96x96 --maxShapes=images:64x3x96x96 --saveEngine=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_int8.plan
trtexec --loadEngine=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_fp32.plan --shapes=images:32x3x96x96
trtexec --loadEngine=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_fp16.plan --shapes=images:32x3x96x96
trtexec --loadEngine=examples/capstone-mobilenetv2-gtsrb/outputs/mobilenetv2_int8.plan --shapes=images:32x3x96x96
```

TensorRT engines are specific to the compatible GPU and software environment. The build and load commands report performance; `trtexec` does not calculate GTSRB accuracy. TensorRT versions before 11 use different precision-build options.

## Limitations

This trains a reduced-width, 96×96 GTSRB classifier from scratch. It is an architecture implementation exercise, not a reproduction of the paper's ImageNet experiment. GTSRB contains cropped signs; the model does not locate signs in full road images. Do not interpret this project as a road-safety system.
