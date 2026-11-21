# MNIST Vision Transformer capstone

- **Created:** 2026-11-21
- **Status:** Implemented
- **Description:** Add a standalone, visual train-to-inference capstone that builds a small Vision Transformer on MNIST with PyTorch.

## Key decisions

- Reuse the existing Vision Transformer lesson for conceptual prerequisites and make the capstone a practical step-by-step implementation. A separate full Transformer course is not required for this learning goal.
- Use 7 × 7 patches on 28 × 28 MNIST images, producing 16 patch tokens with 49 grayscale values each. Project to 64 features and prepend a learned class token with learned position embeddings.
- Use PyTorch's reference `TransformerEncoderLayer` and `TransformerEncoder` with two layers, four heads, and batch-first tensors. Initialize cloned encoder-layer matrix parameters independently as the official API documentation recommends.
- Split only MNIST's official training set into training and validation data. Choose the checkpoint by validation accuracy, then visualize predictions from a reproducibly shuffled batch of the official test split.
- Keep the model, training, and inference in a small shared model module and two scripts. Save weights, learning curves, and inference output under the example's `outputs/` directory.
- Describe the result as a teaching model. Do not imply that a small ViT is an optimal MNIST model or comparable to large pretrained ViTs.

## Related page and examples

- [Capstone guide](../../docs/capstone-mnist-vit/index.md)
- [Shared model](../../examples/capstone-mnist-vit/mnist_vit.py)
- [Training script](../../examples/capstone-mnist-vit/train_vit_mnist.py)
- [Inference script](../../examples/capstone-mnist-vit/infer_vit_mnist.py)
- [Run guide](../../examples/capstone-mnist-vit/README.md)
- [Lesson 11: Vision Transformers](../../docs/modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md)

## Artifact flow

```mermaid
flowchart LR
    A[MNIST image] --> B[Split into 16 patches]
    B --> C[Project patches and add class and position tokens]
    C --> D[Two-layer Transformer encoder]
    D --> E[Ten digit logits]
    E --> F[Loss and training updates]
    F --> G[Best validation checkpoint]
    G --> H[PyTorch test-batch inference]
    H --> I[Prediction grid PNG]
```
