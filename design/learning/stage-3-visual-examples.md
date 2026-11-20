# Stage 3 visual examples

- **Created:** 2026-11-20
- **Status:** Implemented
- **Description:** Plan three small, visual demonstrations for the Modern Computer Vision lessons.

## Key decisions

- Keep each lesson focused on its syllabus objective: patch attention, label-free view matching, or object localization.
- Use small synthetic or in-memory examples so demonstrations run without pretrained weights, large downloads, or specialized hardware. Show ViT patch attention, contrastive self-supervision alongside a conceptual DINO teacher, and color-based object localization.
- Label toy examples clearly and describe where they differ from production ViT, DINO, and YOLO systems.
- Generate plots from runnable scripts, then embed the generated results in each lesson so readers can compare their own outputs.
- Add one README per demo with exact repository-root commands, expected outputs, and limitations.

## Related lessons and examples

- [Lesson 11 — Vision Transformers](../../docs/modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md)
- [Lesson 12 — Self-Supervised Learning](../../docs/modules/03-modern-computer-vision/lesson-12-self-supervised-learning/index.md)
- [Lesson 13 — Object Detection](../../docs/modules/03-modern-computer-vision/lesson-13-object-detection/index.md)
- Runnable examples will live under `examples/03-modern-computer-vision/`.
