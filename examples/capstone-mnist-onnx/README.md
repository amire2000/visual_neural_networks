# MNIST MLP to ONNX

Train a fully connected MLP, export its weights and computation to ONNX, and use ONNX Runtime to classify a shuffled batch from MNIST's official test set.

## Prerequisites

Use PyTorch 2.6 or newer in the course `.venv`. Complete [Setup Lab](../../docs/lesson-00-setup-lab/index.md) to install PyTorch, TorchVision, and Matplotlib. Then install the CPU ONNX packages from the repository root:

```bash
uv pip install --python .venv/bin/python onnx onnxscript onnxruntime
```

The standard `onnxruntime` package runs on CPU. PyTorch itself may use CUDA for MLP training if your installed PyTorch build supports it.

## Run the project step by step

From the repository root:

```bash
uv run python examples/capstone-mnist-onnx/train_mnist_mlp.py
uv run python examples/capstone-mnist-onnx/infer_torch.py
uv run python examples/capstone-mnist-onnx/export_onnx.py
uv run python examples/capstone-mnist-onnx/infer_onnx.py
uv run python examples/capstone-mnist-onnx/quantize_onnx.py
uv run python examples/capstone-mnist-onnx/infer_onnx.py --quantized
uv run python examples/capstone-mnist-onnx/benchmark_inference.py
```

Training downloads MNIST if it is not already in `data/mnist/`. Inference and export need the checkpoint from training. The PyTorch inference command runs without ONNX packages.

### Optional dynamic int8 inference

Install TorchAO compatible with the PyTorch build in the course environment, then run the same reproducible test batch through the quantized model:

```bash
uv pip install --python .venv/bin/python torchao
uv run python examples/capstone-mnist-onnx/infer_torch.py --quantized
```

## Expected outputs

The `outputs/` directory contains:

- `best_mlp.pth` — PyTorch weights from the best validation epoch.
- `training_curves.png` — training and validation loss and accuracy.
- `torch_predictions.png` — true labels and full-precision PyTorch predictions.
- `torch_quantized_predictions.png` — optional TorchAO dynamic int8 predictions.
- `mnist_mlp.onnx` — exported model with a dynamic batch dimension.
- `mnist_mlp_int8.onnx` — optional dynamic int8 quantized copy of the exported model.
- `onnx_predictions.png` — true labels and ONNX Runtime predictions for 16 test images.
- `onnx_int8_predictions.png` — predictions from the quantized model on the same test batch.
- `mnist_calibration.npz` — representative training images for TensorRT INT8 calibration.
- `mnist_fp32.plan`, `mnist_fp16.plan`, `mnist_int8.plan` — optional TensorRT engines built for the local GPU and TensorRT environment.

`benchmark_inference.py` prints a Markdown table comparing model load time, warm average batch inference time, milliseconds per image, throughput, and batch accuracy for PyTorch FP32, ONNX FP32, and ONNX int8. It requires all three model artifacts, uses one CPU thread, and excludes dataset loading and preprocessing from inference timing. Results vary by machine.

The export step checks the ONNX graph and asserts that ONNX Runtime scores match PyTorch scores within a small numeric tolerance. The quantization step also checks its output graph. Both ONNX inference commands use the same reproducible test batch and print its accuracy; this is not a full test-set report. Quantization may change predictions and does not guarantee faster execution for this small model.

## Limitations

This is a small, fully connected MNIST classifier. It does not preserve the image's two-dimensional neighborhood structure explicitly. ONNX Runtime uses CPU in this example. The displayed batch is for visual inspection, not a reliable full-dataset benchmark.

## Optional TensorRT 11 workflow (NVIDIA GPU)

This path uses the trained MLP exported above. It requires a compatible NVIDIA driver, CUDA/TensorRT 11 installation (`trtexec` on `PATH`), and NVIDIA Model Optimizer. TensorRT engines are hardware and version specific. The TensorRT 11 strongly typed workflow converts precision in ONNX first; older TensorRT releases use different builder flags.

```bash
uv run python examples/capstone-mnist-onnx/prepare_calibration_data.py
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com 'nvidia-modelopt[onnx]'
uv run python -m modelopt.onnx.autocast --onnx_path examples/capstone-mnist-onnx/outputs/mnist_mlp.onnx --output_path examples/capstone-mnist-onnx/outputs/mnist_mlp_fp16.onnx
uv run python -m modelopt.onnx.quantization --onnx_path examples/capstone-mnist-onnx/outputs/mnist_mlp.onnx --quantize_mode int8 --calibration_data_path examples/capstone-mnist-onnx/outputs/mnist_calibration.npz --output_path examples/capstone-mnist-onnx/outputs/mnist_mlp_int8_qdq.onnx
trtexec --onnx=examples/capstone-mnist-onnx/outputs/mnist_mlp.onnx --saveEngine=examples/capstone-mnist-onnx/outputs/mnist_fp32.plan
trtexec --onnx=examples/capstone-mnist-onnx/outputs/mnist_mlp_fp16.onnx --saveEngine=examples/capstone-mnist-onnx/outputs/mnist_fp16.plan
trtexec --onnx=examples/capstone-mnist-onnx/outputs/mnist_mlp_int8_qdq.onnx --saveEngine=examples/capstone-mnist-onnx/outputs/mnist_int8.plan
trtexec --loadEngine=examples/capstone-mnist-onnx/outputs/mnist_fp32.plan
trtexec --loadEngine=examples/capstone-mnist-onnx/outputs/mnist_fp16.plan
trtexec --loadEngine=examples/capstone-mnist-onnx/outputs/mnist_int8.plan
```

The last three commands report TensorRT engine startup and inference performance. Compare their build and runtime numbers on the same GPU; engine building is an offline cost. Use the existing ONNX Runtime benchmark for a portable FP32/int8 CPU baseline. Check accuracy on the MNIST test set before deployment: `trtexec` performance timing does not report task accuracy. The calibration script samples 256 normalized images from the training split and saves them under the ONNX input name `images`.

INT8 maps a real value `x` to an integer approximately as `q = clip(round(x / scale) + zero_point, -128, 127)`, then reconstructs it as `x_hat = scale * (q - zero_point)`. Calibration runs representative inputs to choose useful scales; values outside the observed range clip, and rounding/clipping cause approximation error. Symmetric signed INT8 commonly uses zero point 0. The Q/DQ nodes carry the chosen scales so TensorRT knows where to quantize and dequantize.
