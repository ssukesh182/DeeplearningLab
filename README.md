# Deep Learning Laboratory

A collection of five Deep Learning laboratory experiments covering the progression from a basic perceptron to convolutional neural networks, transfer learning, regularization, optimization and model selection.

## Labs Overview

| Lab | Topic | Dataset / Task |
|---|---|---|
| Lab 1 | Single Layer Perceptron | Banknote Authentication, AND/OR/NOT |
| Lab 2 | Multi-Layer Perceptron | Fashion-MNIST, XOR |
| Lab 3 | Convolutional Neural Networks | CIFAR-10 |
| Lab 4 | Transfer Learning with VGG16 | CIFAR-10 |
| Lab 5 | CNN Training and Model Selection | Oxford-IIIT Pet |

---

# Lab 1: Single Layer Perceptron

## Topics

- Perceptron architecture and learning rule
- Binary classification
- Feature normalization
- Exploratory data analysis
- Model evaluation
- Learning rate analysis
- Weight and bias evolution
- Logic gates

## Dataset

**Banknote Authentication Dataset**

- 1,372 samples
- 4 numerical features
- 2 classes
- 1,097 training samples
- 275 testing samples
- Features: Variance, Skewness, Curtosis and Entropy

The features are normalized using `StandardScaler`.

## Perceptron

The weighted sum is:

```text
z = w^T x + b
```

Step activation:

```text
f(z) = 1, if z >= 0
       0, if z < 0
```

Update rule:

```text
w = w + eta * e * x
b = b + eta * e
```

where:

```text
e = y - y_hat
```

## Results

| Metric | Custom Perceptron | Scikit-Learn |
|---|---:|---:|
| Accuracy | 99.27% | 100.00% |
| Precision | 100.00% | 100.00% |
| Recall | 98.36% | 100.00% |
| F1 Score | 99.17% | 100.00% |

The custom implementation achieved 99.27% accuracy.

Different learning rates were also evaluated, with `0.0001` giving the highest reported accuracy of 99.27%.

## Logic Gates

A perceptron successfully learns:

- AND
- OR
- NOT

XOR cannot be learned by a single-layer perceptron because it is not linearly separable.

---

# Lab 2: Multi-Layer Perceptron

## Topics

- Multi-Layer Perceptrons
- Multi-class classification
- ReLU and Softmax
- Cross-entropy loss
- Adam optimizer
- Model evaluation
- Hyperparameter optimization
- Cross-validation
- XOR using an MLP

## Dataset

**Fashion-MNIST**

- 60,000 training images
- 10,000 testing images
- 10 classes
- 28 x 28 grayscale images

Images are normalized to `[0, 1]` and flattened into 784-dimensional vectors.

## Baseline Architecture

```text
Input: 784
   |
Dense: 128, ReLU
   |
Dense: 64, ReLU
   |
Dense: 10, Softmax
```

Configuration:

```text
Optimizer = Adam
Learning Rate = 0.001
Batch Size = 32
Epochs = 20
Loss = Sparse Categorical Cross Entropy
```

## Results

| Metric | Baseline | Optimized |
|---|---:|---:|
| Accuracy | 88.85% | 87.67% |
| Precision | 88.93% | 87.67% |
| Recall | 88.85% | 87.67% |
| F1 Score | 88.76% | 87.56% |
| Training Time | 179.96 s | 51.80 s |

Randomized Search explored:

- Number of hidden layers
- Number of neurons
- Activation function
- Optimizer
- Learning rate
- Batch size
- Epochs
- Dropout

The optimized configuration reduced training time substantially, although its test metrics were slightly lower than the baseline.

## XOR

A single-layer perceptron fails to solve XOR.

An MLP with:

```text
2 inputs
   |
4 hidden neurons, Sigmoid
   |
1 output neuron, Sigmoid
```

correctly learned all four XOR combinations.

After 10,000 epochs, the reported MSE was `0.0023`.

---

# Lab 3: Convolutional Neural Networks

## Topics

- Convolution
- Kernel operations
- Stride and padding
- Feature maps
- Max pooling and average pooling
- CNN parameter calculation
- CIFAR-10 classification
- Filter comparison
- Training and validation analysis

## Dataset

**CIFAR-10**

- 50,000 training images
- 10,000 testing images
- 10 classes
- 32 x 32 RGB images

## Convolution

The output size is calculated using:

```text
Output Size = (N - F + 2P) / S + 1
```

Examples with valid padding and stride 1:

| Kernel | Output |
|---|---|
| 3 x 3 | 30 x 30 |
| 5 x 5 | 28 x 28 |
| 7 x 7 | 26 x 26 |

The experiment also studies common kernels including:

- Identity
- Sobel-X
- Sobel-Y
- Laplacian
- Sharpen
- Box Blur
- Gaussian Blur
- Emboss
- Outline
- Motion Blur

## CNN Architecture

```text
Input
   |
Conv2D(32, 3 x 3)
   |
MaxPool2D(2 x 2)
   |
Conv2D(64, 3 x 3)
   |
MaxPool2D(2 x 2)
   |
Flatten
   |
Dense(64)
   |
Output
```

## Results

The reported model achieved approximately:

```text
Training Accuracy = 86%
Test Accuracy = 68.78%
```

Other reported metrics:

```text
Precision = 69%
Recall    = 68.7%
F1 Score  = 68.7%
```

The original architecture contains `167,562` trainable parameters.

A reduced-filter version using 16 and 32 filters contains `79,530` parameters.

The experiment also compares Max Pooling vs Average Pooling and different numbers of convolution filters.

---

# Lab 4: Transfer Learning with VGG16

## Topics

- CNN architecture evolution
- LeNet-5
- AlexNet
- VGG16
- GoogleNet
- ResNet
- Dilated convolution
- Transpose convolution
- Transfer learning
- Feature extraction
- Fine tuning
- Hyperparameter experiments

## CNN Architecture Evolution

| Model | Depth | Parameters | Main Idea |
|---|---:|---:|---|
| LeNet-5 | 7 | 60K | Early CNN architecture |
| AlexNet | 8 | 61M | ReLU, dropout and GPU training |
| VGG16 | 16 | 138M | Deep network with 3 x 3 filters |
| GoogleNet | 22 | 6.8M | Inception modules |
| ResNet50 | 50 | 25.6M | Residual connections |

## Transfer Learning

VGG16 pretrained on ImageNet is adapted for CIFAR-10.

```text
VGG16 Base
   |
Global Average Pooling
   |
Dense(256, ReLU)
   |
Dense(10, Softmax)
```

Initially, the convolutional base is frozen and only the new classifier is trained.

## Fine Tuning

The final convolutional block is then unfrozen and trained with the classifier.

| Stage | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| Feature Extraction | 70.17% | 62.17% |
| Fine Tuning | 85.26% | 72.17% |

The reported test accuracy after fine tuning was `72.17%`.

## Hyperparameter Experiments

The experiment compares learning rates and dense layer sizes, followed by optimizer and freezing-strategy comparisons.

The reported best configuration was:

```text
Learning Rate = 0.001
Dense Units = 256
Optimizer = Adam
Freezing = Partial
Test Accuracy = 73.82%
```

The best model reached:

```text
Training Accuracy = 87.52%
Testing Accuracy = 73.82%
Weighted Precision = 75.45%
Weighted Recall = 73.82%
Weighted F1 Score = 73.83%
```

---

# Lab 5: CNN Training, Regularization, Optimization and Model Selection

## Topics

- Weight initialization
- Regularization
- L2 regularization
- Batch Normalization
- Dropout
- Optimization algorithms
- CNN hyperparameter tuning
- MobileNetV2 transfer learning
- Fine tuning
- 5-fold cross-validation
- Final model evaluation

## Dataset

**Oxford-IIIT Pet Dataset**

- 37 cat and dog breeds
- RGB images
- Resized to 224 x 224 x 3
- MobileNetV2 pretrained on ImageNet

The test set remains untouched during hyperparameter selection.

## MobileNetV2

MobileNetV2 is used as the pretrained feature extractor.

Important components include:

- Depthwise convolution
- Pointwise 1 x 1 convolution
- Inverted residual blocks
- Linear bottlenecks
- Batch Normalization
- ReLU6

## Regularization

The experiment evaluates:

- No regularization
- L2 regularization
- Batch Normalization
- Dropout

The plots compare training and validation accuracy/loss to study overfitting and generalization.

## Optimizer Comparison

| Optimizer | Final Loss | Best Validation Accuracy | Time |
|---|---:|---:|---:|
| SGD | 2.7655 | 29.40% | 636.09 s |
| Momentum | 0.5602 | 85.00% | 653.26 s |
| RMSProp | 0.0849 | 89.80% | 588.52 s |
| Adam | 0.1082 | 88.00% | 652.53 s |

## Hyperparameter Tuning

Parameters investigated:

```text
Learning Rate: 0.001, 0.0001
Batch Size: 16, 32, 64
Dropout: 0, 0.25, 0.5
Optimizer: SGD, Momentum, RMSProp, Adam
Fine-Tuning Learning Rate: 10^-5
```

Reported results:

```text
Learning Rate 0.001  -> 89.8% validation accuracy
Learning Rate 0.0001 -> 80.4%
```

Batch sizes produced similar validation accuracy, with batch size 32 reaching approximately 89.2%.

## Transfer Learning

Two strategies are compared.

### Feature Extraction

```text
MobileNetV2
    |
Freeze Base
    |
Train Classifier
```

### Fine Tuning

```text
MobileNetV2
    |
Unfreeze Selected Layers
    |
Small Learning Rate
    |
Fine Tune
```

Fine tuning can improve adaptation to the target dataset, but requires substantially more computation.

## 5-Fold Cross-Validation

Three configurations were evaluated:

| Config | Dropout | Optimizer | Learning Rate | Mean ± SD |
|---|---:|---|---:|---:|
| C1 | 0.0 | Adam | 0.001 | 85.9 ± 1.62% |
| C2 | 0.25 | Adam | 0.001 | 84.4 ± 2.62% |
| C3 | 0.5 | Adam | 0.0001 | 16.3 ± 5.78% |

C1 was selected for final evaluation based on the reported balance of cross-validation accuracy, variability and computational cost.

## Final Evaluation

| Metric | Result |
|---|---:|
| Mean CV Accuracy | 85.90% |
| CV Standard Deviation | 1.62% |
| Test Accuracy | 89.00% |
| Precision | 90.22% |
| Recall | 89.00% |
| F1 Score | 88.81% |
| Training Time | 670.86 s |
| Parameters | 2,305,381 |

## Overall Comparison

| Configuration | CV Accuracy | SD | Test Accuracy | Training Time |
|---|---:|---:|---:|---:|
| Baseline | 85.90% | 1.62% | 89.00% | 670.86 s |
| Best Initialization | 88.70% | 1.40% | 90.00% | 706.82 s |
| Best Regularization | 85.90% | 1.62% | 89.00% | 670.86 s |
| Best Optimizer | 85.20% | 2.66% | 88.20% | 637.86 s |
| Best Hyperparameters | 85.90% | 1.62% | 89.00% | 670.86 s |
| Fine-Tuned Model | 80.60% | 2.06% | 89.60% | 1769.49 s |

---

# Concepts Covered Across the Labs

The five experiments progressively cover the main building blocks of deep learning:

```text
Single Layer Perceptron
        |
        v
Multi-Layer Perceptron
        |
        v
Convolutional Neural Networks
        |
        v
Transfer Learning
        |
        v
Regularization + Optimization
        |
        v
Hyperparameter Tuning
        |
        v
Cross-Validation + Model Selection
```

### Core Topics

- Perceptron learning
- Activation functions
- Backpropagation
- Loss functions
- Optimization
- Convolution
- Pooling
- Feature maps
- CNN architectures
- Regularization
- Batch Normalization
- Dropout
- Transfer learning
- Fine tuning
- Hyperparameter optimization
- Cross-validation
- Model evaluation

### Evaluation Metrics

Across the experiments, model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Loss
- Confusion Matrix
- Training and validation curves
- Cross-validation mean
- Cross-validation standard deviation
- Training time
- Number of parameters

## Requirements

Most experiments use Python and the following libraries:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow
- Keras

Install the common dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

## Repository Structure

```text
DeeplearningLab/
├── Lab 1/
│   ├── *.py
│   └── *.ipynb
├── Lab 2/
│   ├── *.py
│   └── *.ipynb
├── Lab 3/
│   ├── *.py
│   └── *.ipynb
├── Lab 4/
│   ├── *.py
│   └── *.ipynb
├── Lab 5/
│   ├── *.py
│   └── *.ipynb
└── README.md
```

The exact file and folder names may vary depending on the repository contents.

## Source

https://github.com/ssukesh/DeeplearningLab
