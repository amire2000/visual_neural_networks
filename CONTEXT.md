# Visual Neural Networks

## Domain model

- **Image tensor:** Numerical representation of an image, organized by height, width, and channel dimensions.
- **CNN:** A neural network that learns local visual patterns with convolutional filters and combines them into richer features.
- **Depthwise convolution:** A spatial convolution that applies a separate filter to each input channel, reducing computation compared with a full channel-mixing convolution.
- **Inverted residual block:** A block that expands feature channels, applies a depthwise convolution, and projects back to a narrow representation; its skip connects the narrow endpoints when shapes match.
- **Linear bottleneck:** The narrow output projection of a MobileNetV2 block with no activation applied afterward.
- **Width multiplier:** A model-scaling factor that changes the number of channels in network stages.
- **Depthwise convolution:** A spatial convolution that applies a separate filter to each input channel, reducing computation compared with a full channel-mixing convolution.
- **Inverted residual block:** A block that expands feature channels, applies a depthwise convolution, and projects back to a narrow representation; its skip connects the narrow endpoints when shapes match.
- **Linear bottleneck:** The narrow output projection of a MobileNetV2 block with no activation applied afterward.
- **Width multiplier:** A model-scaling factor that changes the number of channels in network stages.
- **Classification:** Assigning an image one label from a fixed set of classes.
- **Detection:** Predicting object classes and locations, commonly represented by bounding boxes.
- **Inference:** Running a trained model to produce predictions.

The course progresses from CNN image classification through modern vision methods to inference and deployment. Its syllabus defines scope and outcomes; lesson content is separate work.
