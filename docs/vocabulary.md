# Course Vocabulary

Short definitions of the key ideas used throughout the course. Select **Learn more** to open the lesson that explains a term in context.

| Term | Short description | Learn more |
|---|---|---|
| Accuracy | The share of predictions that match the correct labels. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Activation | A value produced by a neuron or layer after an operation such as ReLU. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Adam | An optimizer that uses gradients to adjust model parameters during training. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| Adaptive average pooling | A pooling layer that averages each feature map to a chosen output size, often 1 × 1. | [8 - Building Deeper CNNs](modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md) |
| Augmented view | A transformed version of an image used as another view of the same scene. | [12 - Self-Supervised Learning](modules/03-modern-computer-vision/lesson-12-self-supervised-learning/index.md) |
| Backpropagation | The process that calculates how model parameters contributed to the loss. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Batch | A group of images processed together in one model step. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| Batch Normalization (BatchNorm) | A layer that normalizes channel activations to help stabilize training. | [8 - Building Deeper CNNs](modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md) |
| Batch size | The number of images processed together in a batch. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Batch-fill wait | Time a live system waits for enough arriving frames to form a batch. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Bias | A learned value added by a convolution or linear layer. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Bounding box | A rectangle that marks an object's location in an image. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| Calibration | Estimating value ranges so a model can be represented in a lower-precision format such as INT8. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Camera frame interval | Time between incoming frames; at 30 FPS it is about 33.3 ms. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Channel | One image plane, such as red, green, blue, or a learned feature map. | [7 - From MNIST to CIFAR-10](modules/02-modern-cnns/lesson-07-from-mnist-to-cifar10/index.md) |
| Checkpoint | A saved set of model weights, often from the best validation epoch. | [6.1 - TensorBoard](modules/01-cnn-fundamentals/lesson-06-1-tensorboard-dashboard/index.md) |
| CIFAR-10 | A dataset of 60,000 small color images in ten object categories. | [7 - From MNIST to CIFAR-10](modules/02-modern-cnns/lesson-07-from-mnist-to-cifar10/index.md) |
| Class score | A raw value a classifier produces for one possible class. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Classification | Assigning an image one label from a fixed list of classes. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Classifier head | The final layer or layers that turn image features into class scores. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| CNN | A convolutional neural network that learns local visual patterns and combines them into features. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Confidence score | A detector's score for how strongly a predicted class or box is supported. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| Contrastive learning | A self-supervised method that brings features from two views of one image together and separates other examples. | [12 - Self-Supervised Learning](modules/03-modern-computer-vision/lesson-12-self-supervised-learning/index.md) |
| Convolution | An operation that applies a small learned filter across local image regions. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Convolution filter (kernel) | A small grid of learned weights that responds to patterns in an image region. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| CPU | A general-purpose processor that can run model inference. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Cross-entropy loss | A common classification loss that measures how well scores favor the correct class. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| Data augmentation | Label-preserving image changes, such as crops or flips, used to vary training examples. | [7 - From MNIST to CIFAR-10](modules/02-modern-cnns/lesson-07-from-mnist-to-cifar10/index.md) |
| Dataset | A collection of examples and, for supervised learning, their labels. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Dependency | A software package a project needs in order to run. | [0 - Setup Lab](lesson-00-setup-lab/index.md) |
| Depth | The number of processing layers in a network. | [8 - Building Deeper CNNs](modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md) |
| Detection | Finding objects in an image and reporting their classes and locations. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| DINO | A self-supervised teacher–student method for learning visual features without class labels. | [12 - Self-Supervised Learning](modules/03-modern-computer-vision/lesson-12-self-supervised-learning/index.md) |
| Edge device | A nearby computer that processes camera data close to where it is captured. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Embedding | A vector representation of an image patch or other input. | [11 - Vision Transformers](modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md) |
| Epoch | One complete pass through the training examples. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| Feature | A learned representation that helps a model distinguish patterns or classes. | [5 - What CNNs Learn](modules/01-cnn-fundamentals/lesson-05-what-cnn-learns/index.md) |
| Feature extractor | The layers that turn an image into useful intermediate features. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Feature map | A grid showing where and how strongly a filter responds to an image. | [5 - What CNNs Learn](modules/01-cnn-fundamentals/lesson-05-what-cnn-learns/index.md) |
| Fine-tuning | Continuing training on a new task while updating selected pretrained layers. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Flatten | Reshaping feature maps into a vector for a classifier. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Forward pass | Sending an input through a model to produce its output. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| FP16 | A 16-bit floating-point format that uses less storage than FP32. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| FP32 | A 32-bit floating-point format commonly used as a model precision baseline. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| FPS (frames per second) | A throughput measure counting how many frames a system processes each second. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Frame rate | How many frames a camera sends per second, commonly expressed in FPS. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Frozen backbone | A pretrained feature extractor whose weights are kept fixed while another part trains. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Generalization | A model's ability to perform well on examples it did not train on. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Global average pooling | Averaging each feature map into one value before classification. | [8 - Building Deeper CNNs](modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md) |
| Gradient | A value that indicates how changing a parameter affects the loss locally. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Graphics processing unit (GPU) | A processor commonly used to accelerate neural-network training and inference. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Grayscale | An image representation with one brightness channel instead of RGB color channels. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Histogram | A chart that groups values into ranges and shows how many fall in each range. | [6.1 - TensorBoard](modules/01-cnn-fundamentals/lesson-06-1-tensorboard-dashboard/index.md) |
| Image tensor | An image represented by numerical values arranged by channel, height, and width. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| ImageNet | A large labeled image dataset commonly used to pretrain vision models. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Inference | Using a model's learned weights to make predictions without updating them. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| INT8 | An 8-bit integer format that can reduce model storage but may affect results. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Intersection over Union (IoU) | A measure of overlap between a predicted box and a reference box. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| Kernel | Another name for a convolution filter: a small grid of learned weights. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Label | The correct class or target associated with a training example. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Latency | Time from receiving an input to producing its result. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Learning curve | A graph of a training or validation measure over epochs. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Learning rate | A setting that controls the size of optimizer updates to model parameters. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Linear layer | A learned mapping from input features to output scores. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Linear probing | Training a simple classifier on fixed pretrained features. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Localization | Predicting where an object appears in an image, often with a bounding box. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| Logit | A raw class score produced before converting scores to probabilities. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Loss | A value that measures how poorly a model's predictions match the target. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Matplotlib | A Python library used in the course to display images and plots. | [0 - Setup Lab](lesson-00-setup-lab/index.md) |
| Max pooling | Keeping the largest value from each small feature-map region to reduce its size. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| MNIST | A dataset of 70,000 small grayscale images of handwritten digits. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Multilayer perceptron (MLP) | A network of fully connected layers that transforms input values into class scores. | [MNIST MLP to ONNX capstone](capstone-mnist-onnx/index.md) |
| Model parameter | A learned number in a model, such as a filter weight or bias. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Model size | The storage needed for model weights and related data. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Non-maximum suppression (NMS) | A detection step that removes duplicate boxes around the same object. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| Normalization | Scaling image values into the range or distribution expected by a model. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| ONNX | A model exchange format that stores a model's computation and learned parameters. | [MNIST MLP to ONNX capstone](capstone-mnist-onnx/index.md) |
| ONNX Runtime | A runtime that loads and executes models saved in ONNX format. | [MNIST MLP to ONNX capstone](capstone-mnist-onnx/index.md) |
| Object detection | Predicting both what objects appear in an image and where they are. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
| Optimizer | A method that uses gradients to update model parameters and reduce loss. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Overfitting | When a model fits its training examples well but performs worse on new examples. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Padding | Extra pixels added around an image before convolution, often to preserve its width and height. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Parameter | A learned value in a model; weights and biases are common examples. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Patch | A small rectangular region into which a Vision Transformer divides an image. | [11 - Vision Transformers](modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md) |
| Patch embedding | A vector representation created from an image patch for a Transformer. | [11 - Vision Transformers](modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md) |
| Pixel | One location in an image, represented by one or more brightness or color values. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Precision | The number format used for model values, such as FP32, FP16, or INT8. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Prediction | A model's output for an input image, such as a class score or detected box. | [1 - How Neural Networks See Images](modules/01-cnn-fundamentals/lesson-01-how-neural-networks-see-images/index.md) |
| Preprocessing | Resizing, normalizing, or otherwise preparing an image before model inference. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Pretrained model | A model whose parameters were already learned on a source task or dataset. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Pretraining | Training a model on a source task before adapting it to a new task. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Projection shortcut | A learned shortcut, often a 1 × 1 convolution, that adjusts tensor shape before a residual addition. | [9 - Understanding ResNet](modules/02-modern-cnns/lesson-09-understanding-resnet/index.md) |
| PyTorch | The Python framework used in the course to build and train neural networks. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| Quantization | Representing model values with fewer bits, often to reduce storage or speed inference. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Receptive field | The part of the input image that can affect one feature activation. | [8 - Building Deeper CNNs](modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md) |
| ReLU | An activation that replaces negative values with zero and keeps positive values. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Residual block | A block that adds a learned change to its input features. | [9 - Understanding ResNet](modules/02-modern-cnns/lesson-09-understanding-resnet/index.md) |
| ResNet | A family of CNNs that uses residual blocks and skip connections. | [9 - Understanding ResNet](modules/02-modern-cnns/lesson-09-understanding-resnet/index.md) |
| RGB | The red, green, and blue channels used to represent a color image. | [7 - From MNIST to CIFAR-10](modules/02-modern-cnns/lesson-07-from-mnist-to-cifar10/index.md) |
| Scalar | A single logged number, such as loss or accuracy, tracked over training steps. | [6.1 - TensorBoard](modules/01-cnn-fundamentals/lesson-06-1-tensorboard-dashboard/index.md) |
| Self-attention | A mechanism that lets image tokens compare and share information with one another. | [11 - Vision Transformers](modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md) |
| Self-supervised learning | Learning useful representations from data without manually supplied class labels. | [12 - Self-Supervised Learning](modules/03-modern-computer-vision/lesson-12-self-supervised-learning/index.md) |
| Skip connection | A direct path that carries features around a group of layers. | [9 - Understanding ResNet](modules/02-modern-cnns/lesson-09-understanding-resnet/index.md) |
| Softmax | A function that converts class scores into probabilities that sum to one. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Stride | How far a convolution or pooling window moves between positions. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| Student–teacher model | A learning setup where a teacher model guides a student model's predictions. | [12 - Self-Supervised Learning](modules/03-modern-computer-vision/lesson-12-self-supervised-learning/index.md) |
| Tensor | A multi-dimensional array used to hold image data, features, or model values. | [3 - CNN Layer Transformations](modules/01-cnn-fundamentals/lesson-03-how-cnn-layers-transform-images/index.md) |
| TensorBoard | A dashboard for viewing logged training metrics, images, and weight histograms. | [6.1 - TensorBoard](modules/01-cnn-fundamentals/lesson-06-1-tensorboard-dashboard/index.md) |
| TensorRT | NVIDIA's inference optimization toolkit for supported NVIDIA GPUs. | [14 - Model Optimization](modules/04-practical-computer-vision/lesson-14-model-optimization/index.md) |
| Test set | Examples held aside for a final model evaluation after design decisions are complete. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Throughput | The number of frames or examples a system processes per unit of time. | [15 - Real-Time Computer Vision](modules/04-practical-computer-vision/lesson-15-real-time-computer-vision/index.md) |
| Token | A vector in the sequence that a Vision Transformer processes; image patches become tokens. | [11 - Vision Transformers](modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md) |
| TorchVision | PyTorch's vision library, including image datasets, transforms, and pretrained models. | [7 - From MNIST to CIFAR-10](modules/02-modern-cnns/lesson-07-from-mnist-to-cifar10/index.md) |
| TorchAO | PyTorch's library for model quantization and related optimization workflows. | [MNIST MLP to ONNX capstone](capstone-mnist-onnx/index.md) |
| Training | Updating model parameters from labeled examples to reduce loss. | [4 - Build a CNN in PyTorch](modules/01-cnn-fundamentals/lesson-04-build-first-cnn-pytorch/index.md) |
| Training set | Examples used to calculate loss and update model parameters. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Transfer learning | Reusing a model trained on one task and adapting it to another. | [10 - Transfer Learning](modules/02-modern-cnns/lesson-10-transfer-learning/index.md) |
| Underfitting | When a model has not learned patterns well enough to perform on training or new examples. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| uv | A tool used in the setup lesson to create the Python environment and run the course. | [0 - Setup Lab](lesson-00-setup-lab/index.md) |
| Validation set | Held-out examples used during development to compare models and choose settings. | [6 - Training and Overfitting](modules/01-cnn-fundamentals/lesson-06-training-overfitting-improve-accuracy/index.md) |
| Virtual environment | An isolated Python environment for a project's installed packages. | [0 - Setup Lab](lesson-00-setup-lab/index.md) |
| Vision Transformer (ViT) | A vision model that represents image patches as tokens and processes them with Transformer layers. | [11 - Vision Transformers](modules/03-modern-computer-vision/lesson-11-vision-transformers/index.md) |
| Weight | A learned numerical value that controls how strongly an input contributes to a model response. | [2 - How Filters Learn Patterns](modules/01-cnn-fundamentals/lesson-02-how-cnn-filters-learn-patterns/index.md) |
| Width | The number of feature channels available in a network layer. | [8 - Building Deeper CNNs](modules/02-modern-cnns/lesson-08-building-deeper-cnns/index.md) |
| YOLO | A family of object detectors that predicts candidate boxes and class scores in one model pass. | [13 - Object Detection](modules/03-modern-computer-vision/lesson-13-object-detection/index.md) |
