# Stage 4 deployment benchmarks

- **Created:** 2026-11-20
- **Status:** Implemented
- **Description:** Plan practical, visual examples for model optimization and real-time vision deployment.

## Key decisions

- Use one camera-on-a-conveyor parcel-inspection scenario across both lessons.
- Use a small TorchVision MobileNetV3-Small workload, generated input frames, and the course `.venv`; avoid downloads and pretrained-model setup.
- Lesson 14 measures CPU/GPU FP32 and available GPU FP16 latency, compares estimated FP32/FP16/INT8 parameter storage, and reports FP16 output drift. Clearly label INT8 storage as an estimate and TensorRT as a conceptual option rather than a measured path.
- Lesson 15 measures a camera-to-model pipeline across two input resolutions and several batch sizes. Compare measured throughput with a 30 FPS stream and include the average batch-fill wait in the response-latency graph.
- Generate graph PNGs from the runnable scripts and embed the outputs in the lesson pages; explain that measurements vary by hardware and random weights are used for speed comparisons only.
- Give each lesson example a README with exact commands, expected output, and limitations.

## Related lessons and examples

- [Lesson 14 — Model Optimization](../../docs/modules/04-practical-computer-vision/lesson-14-model-optimization/index.md)
- [Lesson 15 — Real-Time Computer Vision](../../docs/modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md)
- Runnable examples will live under `examples/04-practical-computer-vision/`.
