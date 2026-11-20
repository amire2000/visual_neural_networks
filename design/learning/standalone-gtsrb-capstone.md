# Standalone GTSRB capstone

- **Created:** 2026-11-20
- **Status:** Implemented
- **Description:** Add an unnumbered, end-to-end project where learners design and train an image classifier for German traffic signs.

## Key decisions

- Use TorchVision's GTSRB classification dataset with all 43 traffic-sign classes. The project is a standalone capstone after Stage 4, so the course remains 15 numbered lessons plus the optional Lesson 6.1.
- Require a compact custom CNN and offer pretrained ResNet-18 as an optional comparison. Both predict the same 43 classes and use the same train, validation, and test protocol.
- Group training images by their recording-track filename prefix before creating a deterministic, per-class train/validation split. This prevents adjacent images from the same track leaking across those sets. Keep the official test split untouched until model selection is complete.
- Resize images to a fixed input size, use modest rotation and translation only for training, and avoid horizontal flips because they can change traffic-sign meaning.
- Save the best validation checkpoint and generate a learning-curve graph, confusion matrix, sample grid, and misclassified-image gallery so learners can interpret results.
- Store downloaded data and generated artifacts under ignored `data/` and example `outputs/` paths.

## Related pages and example

- [Capstone guide](../../docs/capstone-gtsrb/index.md)
- [Training script](../../examples/capstone-gtsrb/train_gtsrb.py)
- [Example run guide](../../examples/capstone-gtsrb/README.md)

## Data split flow

```mermaid
flowchart LR
    A[Official GTSRB training images] --> B[Group images by recording track]
    B --> C[Per-class train and validation split]
    C --> D[Train model and select checkpoint]
    D --> E[Evaluate once on official test set]
```
