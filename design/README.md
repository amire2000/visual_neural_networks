# Design notes

Design notes record course and implementation decisions that shape more than one lesson.

## Learning

- [Stage 3 visual examples](learning/stage-3-visual-examples.md): lesson scope and runnable-demo choices for Vision Transformers, self-supervised learning, and object detection.
- [Stage 2 modern CNN lessons](learning/stage-2-modern-cnn-lessons.md): park-camera scenario and visual examples for deeper CNNs, ResNet, and transfer learning.
- [Stage 4 deployment benchmarks](learning/stage-4-deployment-benchmarks.md): parcel-camera scenario and measured precision, resolution, and batching comparisons.
- [Standalone GTSRB capstone](learning/standalone-gtsrb-capstone.md): end-to-end traffic-sign classifier with a custom CNN and optional pretrained ResNet-18.
- [MNIST MLP to ONNX capstone](learning/mnist-mlp-onnx-capstone.md): train a non-convolutional baseline, run PyTorch and ONNX inference, and optionally quantize linear layers.
- [MNIST TensorRT optimization path](learning/mnist-tensorrt-optimization.md): extend the trained MNIST ONNX example with TensorRT FP32, FP16, calibrated INT8 engines, and comparable measurements.
- [MNIST Vision Transformer capstone](learning/mnist-vit-capstone.md): train a compact patch-based Vision Transformer on MNIST and run visual PyTorch inference.
- [MobileNetV2 paper-to-GTSRB capstone](learning/mobilenetv2-paper-to-gtsrb-capstone.md): translate the MobileNetV2 paper into a small GTSRB implementation and compare ONNX precision/runtime options.
- [ResNet-20 paper-to-code capstone](learning/resnet20-paper-to-code-capstone.md): read the ResNet paper's CIFAR design, implement its operators, and interpret training graphs.
