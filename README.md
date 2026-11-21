# Visual Neural Networks

Visual Neural Networks is a beginner-friendly course on how computers interpret images. It starts with pixels and simple CNNs, then builds toward modern vision methods and real-time inference. The lessons favor visual explanations, intuitive examples, and small amounts of mathematics.

The course uses **MNIST**, **CIFAR-10**, and **PyTorch** to connect concepts to practical image-classification and computer-vision examples.

## What you will learn

- **Stage 1 — CNN fundamentals:** how images become tensors, filters learn patterns, and a CNN is trained and evaluated.
- **Stage 2 — Modern CNNs:** color images, deeper networks, ResNet, and transfer learning.
- **Stage 3 — Modern computer vision:** Vision Transformers, self-supervised learning, and object detection.
- **Stage 4 — Practical computer vision:** model optimization, latency, throughput, and real-time deployment.

There are 15 core lessons, one optional TensorBoard supplement, and standalone [MNIST-to-ONNX](docs/capstone-mnist-onnx/index.md), [MNIST Vision Transformer](docs/capstone-mnist-vit/index.md), and [GTSRB traffic-sign](docs/capstone-gtsrb/index.md) capstones. Lessons include diagrams, animations, quizzes, hands-on activities, and runnable examples. The [vocabulary](docs/vocabulary.md) page summarizes important terms and links back to the lessons.

## Start learning

1. Read the [roadmap](docs/roadmap.md) and [syllabus](docs/syllabus.md).
2. Follow [0 - Setup Lab](docs/lesson-00-setup-lab/index.md) to prepare the environment and download MNIST.
3. Continue with [1 - How Neural Networks See Images](docs/modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md).

## Run the course site

The project requires Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
uv run mkdocs serve
```

Open the local address printed by MkDocs. Stop the server with `Ctrl+C`.

The site dependencies are installed by `uv sync`. The image examples also use PyTorch, TorchVision, and Matplotlib; follow the Setup Lab to install the PyTorch build that matches your computer.

For example, after installing the vision packages, download MNIST and inspect its files with:

```bash
uv run python examples/00-setup/download_mnist.py
```

Example-specific READMEs describe other dataset or pretrained-weight downloads, run commands, expected output, and limitations.

## Project layout

- `docs/` — course pages, syllabus, roadmap, glossary, and lesson visuals.
- `examples/` — runnable Python demonstrations organized by stage and lesson.
- `design/` — notes on course structure and example decisions.

## Build the site

```bash
uv run mkdocs build --strict
```
