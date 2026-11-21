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

`benchmark_inference.py` prints a Markdown table comparing model load time, warm average batch inference time, milliseconds per image, throughput, and batch accuracy for PyTorch FP32, ONNX FP32, and ONNX int8. It requires all three model artifacts, uses one CPU thread, and excludes dataset loading and preprocessing from inference timing. Results vary by machine.

The export step checks the ONNX graph and asserts that ONNX Runtime scores match PyTorch scores within a small numeric tolerance. The quantization step also checks its output graph. Both ONNX inference commands use the same reproducible test batch and print its accuracy; this is not a full test-set report. Quantization may change predictions and does not guarantee faster execution for this small model.

## Limitations

This is a small, fully connected MNIST classifier. It does not preserve the image's two-dimensional neighborhood structure explicitly. ONNX Runtime uses CPU in this example. The displayed batch is for visual inspection, not a reliable full-dataset benchmark.
