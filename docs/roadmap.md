# Roadmap

## Final goal

Build an intuitive understanding of how modern vision models learn from images and what it takes to run them efficiently in real systems.

| Part | Lessons | Module | Dataset / focus | Milestone |
| --- | --- | --- | --- | --- |
| CNN Basics | 1–6 | CNN Fundamentals | MNIST | Understand, train, and evaluate a basic MNIST CNN. |
| Modern CNNs | 7–10 | Understanding Modern CNNs | CIFAR-10 | Understand foundational modern CNN techniques and pretrained-model adaptation. |
| Beyond CNNs | 11–13 | Modern Computer Vision | ViT, self-supervision, detection | Understand the conceptual differences among CNNs, Vision Transformers, self-supervised learning, and object detection. |
| Vision in Practice | 14–15 | Practical Computer Vision | Optimization and deployment | Understand the fundamentals of optimizing and deploying vision models for real-time inference. |

Each part ends with a short **stage review**. Reviews connect ideas across lessons, check understanding, and provide a small application prompt. They are checkpoints, not extra lessons:

- CNN Basics: [review the image-to-prediction workflow](modules/01-cnn-fundamentals/review/index.md).
- Modern CNNs: [compare deeper CNNs, ResNet, and transfer learning](modules/02-modern-cnns/review/index.md).
- Beyond CNNs: [compare architectures and task types](modules/03-modern-computer-vision/review/index.md).
- Vision in Practice: [review inference trade-offs](modules/04-practical-computer-vision/review/index.md).

## Recommended sequence

Before CNN Basics, complete [Lesson 0: Setup Lab](lesson-00-setup-lab/index.md) to prepare the Python environment and download MNIST. Then work through the four parts in order: build the CNN mental model, expand to modern CNNs, explore other vision architectures, and finish with practical inference. Complete the linked review at the end of each part before moving on.

See the [syllabus](syllabus.md) for each lesson's concepts and intended outcome.

## Capstone project

After the four parts, apply the course ideas in standalone projects, separate from the numbered lessons:

- [MNIST MLP to ONNX](capstone-mnist-onnx/index.md): train a fully connected network, export it, and run predictions with ONNX Runtime.
- [MNIST Vision Transformer](capstone-mnist-vit/index.md): build a small ViT, train it on MNIST, and classify unseen digits with PyTorch.
- [German Traffic Sign Recognition](capstone-gtsrb/index.md): design and train a 43-class CNN, then optionally compare it with pretrained ResNet-18.
- [ResNet-20 Paper to Code](capstone-resnet-cifar10/index.md): learn to read an architecture paper, implement its CIFAR network, and interpret training graphs.
- [MobileNetV2 Paper to Deployment](capstone-mobilenetv2-gtsrb/index.md): read the architecture paper, implement and train a small MobileNetV2 on GTSRB, then compare ONNX precision and runtimes.
