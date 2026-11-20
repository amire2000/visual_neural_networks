# Stage 2 — Modern CNN lessons

- **Created:** 2026-11-20
- **Status:** Implemented
- **Description:** Complete Lessons 8–10 with a park-camera scenario, visual explanations, runnable architecture demos, and animated diagrams.

## Key decisions

- Continue the CIFAR-10 dataset focus and use a park trail camera as the recurring scenario. State that CIFAR-10 is a teaching proxy and a deployed camera needs task-specific data.
- Lesson 8 compares a plain classifier head with global average pooling and demonstrates BatchNorm plus feature-map shape changes without a dataset download.
- Lesson 9 demonstrates the residual addition directly with a same-shape block and visualizes the identity path, learned change, and sum.
- Lesson 10 uses an ImageNet-pretrained ResNet-18 as a feature backbone, adapts its classifier to ten CIFAR-10 classes, and compares head-only training with partial fine-tuning. The demo downloads weights once and does not claim CIFAR accuracy.
- Use standalone animated SVG diagrams with static content visible when animation is disabled. Keep each code demo independently runnable and document downloads and limitations.

## Related lessons and examples

- [Lesson 7 — From MNIST to CIFAR-10](../../docs/modules/02-modern-cnns/lesson-07-from-mnist-to-cifar10/index.md)
- [Lesson 8 — Building Deeper CNNs](../../docs/modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md)
- [Lesson 9 — Understanding ResNet](../../docs/modules/02-modern-cnns/lesson-09-understanding-resnet/index.md)
- [Lesson 10 — Transfer Learning](../../docs/modules/02-modern-cnns/lesson-10-transfer-learning/index.md)
