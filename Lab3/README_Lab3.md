# Deep Learning Lab 3: Convolutional Neural Networks

Implementation of Convolutional Neural Networks (CNNs) for image classification using TensorFlow/Keras and the CIFAR-10 dataset.

The experiment covers convolution operations, kernel behaviour, stride and padding, feature map visualization, pooling, CNN parameter calculation, model training and evaluation.

## Overview

This experiment covers:

- Convolution operations
- Convolution output size calculation
- Common image processing kernels
- Stride and padding
- Feature map visualization
- Max pooling and average pooling
- CNN parameter calculation
- CIFAR-10 image classification
- Training and validation analysis
- Confusion matrix analysis
- Comparison of different pooling methods
- Comparison of different numbers of convolution filters

## CNN Architecture

A Convolutional Neural Network is built using convolutional layers, activation functions, pooling layers and fully connected layers.

The convolution operation is:

```text
Y(i, j) = sum over m,n of X(i + m, j + n) K(m, n)
```

where:

- `X` is the input image
- `K` is the kernel or filter
- `Y` is the resulting feature map

The spatial output size of a convolution is calculated using:

```text
Output Size = (N - F + 2P) / S + 1
```

where:

- `N` = input size
- `F` = filter size
- `P` = padding
- `S` = stride

## Common Convolution Kernels

The experiment studies several commonly used image-processing kernels.

| Kernel | Size | Purpose | Application |
|---|---|---|---|
| Identity | 3 x 3 | Preserve original image | Baseline comparison |
| Sobel-X | 3 x 3 | Detect vertical edges | Object detection |
| Sobel-Y | 3 x 3 | Detect horizontal edges | Boundary detection |
| Laplacian | 3 x 3 | Detect edges in all directions | Medical imaging |
| Sharpen | 3 x 3 | Enhance details | Image enhancement |
| Box Blur | 3 x 3 | Smooth image | Noise removal |
| Gaussian Blur | 3 x 3 | Reduce noise | Computer vision |
| Emboss | 3 x 3 | Create 3D effect | Artistic filtering |
| Outline | 3 x 3 | Extract boundaries | Segmentation |
| Motion Blur | 5 x 5 | Simulate motion | Video processing |

## Numerical Examples

### Convolution

For:

```text
X = [1 2 3
     4 5 6
     7 8 9]

K = [1 0
     0 1]
```

The four valid convolution positions produce:

```text
6, 8, 12, 14
```

Resulting feature map:

```text
[6  8
 12 14]
```

### Max Pooling

For the following feature map:

```text
[1 5 2 3
 7 8 1 0
 4 6 9 5
 2 3 1 8]
```

A `2 x 2` max-pooling operation produces:

```text
[8 3
 6 9]
```

### Parameter Calculation

For an input of `32 x 32 x 3`, 16 filters and a `3 x 3` kernel:

```text
Parameters = (3 x 3 x 3 + 1) x 16
           = 448
```

The convolution layer therefore contains 448 trainable parameters.

## Dataset

The experiment uses the CIFAR-10 dataset.

| Parameter | Value |
|---|---:|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Classes | 10 |
| Image Size | 32 x 32 x 3 |
| Image Type | RGB |

### Classes

The ten CIFAR-10 classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The notebook loads the dataset using TensorFlow/Keras and examines sample images and class distributions.

## Experimental Tasks

### 1. Load and Inspect CIFAR-10

The dataset is loaded using TensorFlow/Keras.

The experiment displays sample training images and plots the class distribution for both training and test sets.

### 2. Convolution Layer

Feature map sizes are compared for different kernel sizes using stride 1 and valid padding.

| Kernel | Stride | Feature Map Size |
|---|---:|---:|
| 3 x 3 | 1 | 30 x 30 |
| 5 x 5 | 1 | 28 x 28 |
| 7 x 7 | 1 | 26 x 26 |

With valid padding and stride 1, increasing the kernel size reduces the spatial dimensions of the output.

### 3. Stride and Padding

A `3 x 3` kernel is used to examine different stride and padding configurations.

| Configuration | Input | Output Spatial Size |
|---|---|---|
| Stride 1, valid | 32 x 32 | 30 x 30 |
| Stride 2, valid | 32 x 32 | 15 x 15 |
| Stride 1, same | 32 x 32 | 32 x 32 |

A larger stride reduces the spatial resolution more aggressively. With same padding and stride 1, the spatial dimensions are preserved.

### 4. Feature Map Visualization

A convolutional layer containing 32 filters of size `3 x 3` is applied to a normalized CIFAR-10 image.

The first eight feature maps are visualized to observe the responses of different filters.

Early convolutional filters can respond to structures such as edges, textures and patterns.

### 5. Max Pooling vs Average Pooling

A `2 x 2` pooling window is applied to an input with shape `30 x 30 x 32`.

| Pooling Type | Pool Size | Output Spatial Size |
|---|---|---|
| Max Pooling | 2 x 2 | 15 x 15 |
| Average Pooling | 2 x 2 | 15 x 15 |

Both operations reduce the spatial dimensions from `30 x 30` to `15 x 15`.

The notebook also compares validation accuracy and validation loss when the two pooling methods are used in the CNN.

## CNN Model

The main CNN architecture used in the experiment is:

```text
Input
  |
  v
Conv2D(32, 3 x 3)
  |
  v
MaxPool2D(2 x 2)
  |
  v
Conv2D(64, 3 x 3)
  |
  v
MaxPool2D(2 x 2)
  |
  v
Flatten
  |
  v
Dense(64)
  |
  v
Output
```

The input pixel values are scaled to the range `[0, 1]`.

The model is compiled using:

- Optimizer: Adam
- Loss: Sparse Categorical Cross Entropy
- Batch size: 32
- Epochs: 20

## Parameter Calculation

For the original CNN architecture:

```text
Conv1 = (3 x 3 x 3 + 1) x 32
      = 896

Conv2 = (3 x 3 x 32 + 1) x 64
      = 18,496

Dense = (6 x 6 x 64) x 64 + 64
      = 147,520

Output = (64 + 1) x 10
       = 650
```

Total trainable parameters:

```text
167,562
```

## Model Evaluation

The trained CNN is evaluated on the scaled CIFAR-10 test set.

The experiment generates:

- Test loss
- Test accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix
- Training accuracy curve
- Validation accuracy curve
- Training loss curve
- Validation loss curve

### Reported Results

| Metric | Value |
|---|---:|
| Training Accuracy | 0.86 |
| Validation/Test Accuracy | 0.6878 / 0.69 |
| Precision | 0.69 |
| Recall | 0.687 |
| F1 Score | 0.687 |
| Original Model Parameters | 167,562 |

The confusion matrix shows the number of correctly classified samples along the diagonal and the class confusions in the off-diagonal entries.

## Training and Validation Analysis

The notebook records accuracy and loss over 20 epochs.

Training accuracy increases as the model learns from the training data. The difference between training and validation accuracy provides an indication of generalization and possible overfitting.

Similarly, training loss decreases during training. If validation loss starts increasing while training loss continues to decrease, it can indicate overfitting.

## Additional Experiments

### Average Pooling

An otherwise identical CNN is constructed by replacing the MaxPool2D layers with AveragePooling2D.

Both models are trained for 20 epochs with a batch size of 32, and their validation accuracy and validation loss are compared.

### Number of Convolution Filters

Two CNN configurations are compared:

**Fewer-filter model**

```text
Conv1: 16 filters
Conv2: 32 filters
```

**Original model**

```text
Conv1: 32 filters
Conv2: 64 filters
```

Both models are trained for 20 epochs with a batch size of 32.

The experiment compares their validation curves and training times.

## Fewer-Filter Model Parameters

For the model using 16 filters in the first convolution layer and 32 filters in the second:

```text
Conv1 = (3 x 3 x 3 + 1) x 16
      = 448

Conv2 = (3 x 3 x 16 + 1) x 32
      = 4,640

Dense = (6 x 6 x 32) x 64 + 64
      = 73,792

Output = 650
```

Total trainable parameters:

```text
79,530
```

This is lower than the 167,562 parameters in the original model.

The experiment demonstrates that increasing the number of convolution filters increases the representational capacity and the number of trainable parameters.

## Key Observations

- Convolution uses local receptive fields and shared weights, making it more suitable for image data than fully connected layers.
- Increasing the kernel size with valid padding reduces the spatial dimensions of the feature map.
- Increasing the stride reduces spatial resolution.
- Same padding with stride 1 preserves spatial dimensions.
- Pooling reduces spatial dimensions and computational cost.
- Feature maps show the response of learned filters at different locations in an image.
- CNNs use weight sharing, which reduces the number of parameters compared with fully connected networks.
- The original CNN contains 167,562 trainable parameters.
- The fewer-filter architecture contains 79,530 trainable parameters.
- The reported CNN test accuracy is approximately 69%.
- Training and validation curves can be used to examine model learning and possible overfitting.
- Increasing the number of filters increases the number of trainable parameters and the representational capacity of the model.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## Installation

Clone the repository:

```bash
git clone https://github.com/ssukesh182/DeeplearningLab.git
cd DeeplearningLab
```

Install the required packages:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

Run the notebook or Python implementation included in the repository.

## Project Structure

The repository contains the CNN implementation, CIFAR-10 experiments, visualizations and supporting code.

A typical structure is:

```text
DeeplearningLab/
├── *.ipynb
├── *.py
├── plots/
└── README.md
```

The exact file names may vary depending on the implementation.

## References

1. Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning*.
2. Christopher Bishop, *Pattern Recognition and Machine Learning*.
3. Simon Haykin, *Neural Networks and Learning Machines*.
4. TensorFlow Documentation.
5. CIFAR-10 Dataset Documentation.

## Source Code

The complete implementation is available in the repository:

https://github.com/ssukesh182/DeeplearningLab
