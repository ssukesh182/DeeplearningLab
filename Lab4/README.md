# Deep Learning Lab 4: Transfer Learning with VGG16

Comparative study of deep convolutional neural network architectures and implementation of transfer learning using VGG16 on the CIFAR-10 dataset.

## Overview

This experiment studies the evolution of CNN architectures and demonstrates how a pretrained network can be adapted to a new image classification task.

The experiment covers:

- Evolution of CNN architectures
- LeNet-5, AlexNet, VGG16, GoogleNet and ResNet
- Dilated convolution
- Transpose convolution
- Transfer learning
- VGG16 with ImageNet pretrained weights
- Freezing and unfreezing convolutional layers
- Fine tuning
- Hyperparameter experiments
- CIFAR-10 classification
- Model evaluation using accuracy, precision, recall and F1-score

## CNN Architecture Evolution

| Architecture | Year | Main Contribution |
|---|---:|---|
| LeNet-5 | 1998 | First practical convolutional neural network |
| AlexNet | 2012 | ReLU, dropout and GPU training |
| VGG16 | 2014 | Uniform 3 x 3 convolution filters and increased depth |
| GoogleNet | 2014 | Inception modules for multi-scale feature extraction |
| ResNet | 2015 | Residual learning using skip connections |

## Architecture Comparison

The architectures discussed in the experiment are compared below.

| Model | Depth | Parameters | Main Contribution |
|---|---:|---:|---|
| LeNet-5 | 7 | 60K | First CNN |
| AlexNet | 8 | 61M | ReLU + Dropout |
| VGG16 | 16 | 138M | Deep 3 x 3 filters |
| GoogleNet | 22 | 6.8M | Inception modules |
| ResNet50 | 50 | 25.6M | Residual learning |

## LeNet-5

LeNet-5 was proposed by Yann LeCun in 1998 for handwritten digit recognition.

It uses:

- Convolution layers
- Average pooling layers
- Fully connected layers

Its relatively small number of parameters makes it suitable for tasks such as OCR and handwritten digit recognition.

## AlexNet

AlexNet achieved a major breakthrough in the 2012 ImageNet challenge.

Important features introduced or popularized by the architecture include:

- ReLU activation
- Dropout
- GPU-based training
- Data augmentation

## VGG16

VGG16 uses repeated 3 x 3 convolution filters and increases network depth while maintaining a relatively uniform architecture.

The experiment describes its main advantages as:

- Strong feature extraction
- Simple and uniform architecture
- High classification accuracy

## GoogleNet

GoogleNet introduced the Inception module.

An Inception module performs multiple operations in parallel using different filter sizes, such as:

- 1 x 1 convolution
- 3 x 3 convolution
- 5 x 5 convolution
- Pooling

The resulting feature maps are concatenated to produce multi-scale feature representations.

## ResNet

ResNet addresses optimization difficulties associated with very deep networks using residual or skip connections.

Instead of directly learning:

```text
H(x)
```

a residual block learns:

```text
F(x) = H(x) - x
```

and produces:

```text
Output = F(x) + x
```

Residual connections provide a direct path for information and gradients and help very deep networks train more effectively.

## Dilated Convolution

Dilated convolution, also called atrous convolution, increases the receptive field without proportionally increasing the number of kernel parameters.

For standard convolution:

```text
D = 1
```

For dilated convolution:

```text
D > 1
```

Applications include:

- Semantic segmentation
- Medical image analysis
- Satellite imagery
- Object localization

## Transpose Convolution

Transpose convolution performs learnable upsampling and increases the spatial dimensions of feature maps.

Applications include:

- Image super-resolution
- Autoencoders
- GANs
- Image generation
- Semantic segmentation

## Transfer Learning

Transfer learning uses a pretrained model that has already learned general visual features from a large dataset such as ImageNet.

Instead of training every parameter from scratch, the pretrained convolutional base is adapted to the target classification task.

The workflow used in this experiment is:

```text
ImageNet
   |
   v
VGG16 with pretrained weights
   |
   v
Remove original classifier
   |
   v
Freeze convolutional base
   |
   v
Global Average Pooling
   |
   v
Dense(256, ReLU)
   |
   v
Dense(10, Softmax)
   |
   v
CIFAR-10 classification
   |
   v
Unfreeze last convolution block
   |
   v
Fine tune
```

## Dataset

The experiment uses the CIFAR-10 dataset.

| Parameter | Value |
|---|---:|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Number of Classes | 10 |
| Image Size | 32 x 32 x 3 |
| Input Type | RGB |

### Classes

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

Pixel values are converted to floating point values and normalized to the range `[0, 1]`.

## Experimental Procedure

### Task 1: Dataset Preparation

CIFAR-10 is loaded using TensorFlow/Keras.

The images are normalized by dividing pixel values by 255. Sample images are displayed together with their class names.

The low spatial resolution of CIFAR-10 images makes some classes difficult to distinguish visually.

### Task 2: Transfer Learning

VGG16 is loaded with ImageNet pretrained weights using:

```python
include_top=False
```

The input shape is:

```text
(32, 32, 3)
```

The VGG16 convolutional base is initially frozen.

The custom classification head is:

```text
VGG16 Base
    |
GlobalAveragePooling2D
    |
Dense(256, ReLU)
    |
Dense(10, Softmax)
```

### Model Parameters

| Component | Parameters | Initial Status |
|---|---:|---|
| VGG16 convolutional base | 14,714,688 | Frozen |
| Global Average Pooling | 0 | Trainable |
| Dense 256 ReLU | 131,328 | Trainable |
| Dense 10 Softmax | 2,570 | Trainable |
| Total | 14,848,586 | |

## Initial Training

The model is compiled using:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 10 |
| Loss | Sparse Categorical Crossentropy |
| Metric | Accuracy |

After the initial training:

| Metric | Value |
|---|---:|
| Training Accuracy | 0.7017 |
| Validation Accuracy | 0.6217 |

The pretrained convolutional base provides useful visual features, but the frozen feature extractor does not completely adapt to CIFAR-10.

## Fine Tuning

The last convolutional block of VGG16 is unfrozen while the earlier convolutional blocks remain frozen.

The model is then recompiled and trained for five additional epochs.

### Before vs After Fine Tuning

| Stage | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| Before Fine Tuning | 0.7017 | 0.6217 |
| After Fine Tuning | 0.8526 | 0.7217 |

The validation accuracy increased from 62.17% to 72.17% after fine tuning.

## Model Evaluation

After fine tuning, the model is evaluated on the CIFAR-10 test set.

| Metric | Value |
|---|---:|
| Test Loss | 0.8814 |
| Test Accuracy | 0.7217 |

### Classification Report

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Airplane | 0.78 | 0.81 | 0.80 |
| Automobile | 0.72 | 0.92 | 0.81 |
| Bird | 0.83 | 0.58 | 0.68 |
| Cat | 0.51 | 0.61 | 0.56 |
| Deer | 0.63 | 0.70 | 0.66 |
| Dog | 0.72 | 0.52 | 0.60 |
| Frog | 0.73 | 0.80 | 0.77 |
| Horse | 0.69 | 0.86 | 0.77 |
| Ship | 0.86 | 0.83 | 0.84 |
| Truck | 0.93 | 0.57 | 0.71 |
| Macro Average | 0.74 | 0.72 | 0.72 |
| Weighted Average | 0.74 | 0.72 | 0.72 |

The confusion matrix shows stronger classification for classes such as automobile, airplane, frog, horse and ship. Classes with visually similar features, particularly cat and dog, show more confusion.

## Hyperparameter Study

The experiment studies learning rate and dense layer size while keeping other settings fixed.

### Learning Rate and Dense Layer Experiments

| Experiment | Learning Rate | Dense Units | Test Accuracy |
|---|---:|---:|---:|
| 1 | 0.001 | 128 | 0.6117 |
| 2 | 0.001 | 256 | 0.6134 |
| 3 | 0.0001 | 128 | 0.5815 |
| 4 | 0.0001 | 256 | 0.5872 |

The experiment is then expanded to compare Adam and SGD and different freezing strategies.

### Optimizer and Freezing Experiments

| Experiment | Optimizer | Frozen Layers | Test Accuracy |
|---|---|---|---:|
| 5 | Adam | All | 0.6219 |
| 6 | Adam | Partial | 0.7382 |
| 7 | SGD | All | 0.4934 |
| 8 | SGD | Partial | 0.7124 |

The reported best-performing configuration is Experiment 6:

```text
Learning Rate = 0.001
Dense Units = 256
Optimizer = Adam
Frozen Layers = Partial
Test Accuracy = 73.82%
```

## Best Model

Experiment 6 is used as the best-performing model for the final training and validation analysis.

### Performance

| Metric | Value |
|---|---:|
| Training Accuracy | 0.8752 |
| Testing Accuracy | 0.7382 |
| Weighted Precision | 0.7545 |
| Weighted Recall | 0.7382 |
| Weighted F1 Score | 0.7383 |
| Total Parameters | 14,848,586 |
| Training Time | 200 s |

The training accuracy reaches 87.52%, while validation/test accuracy reaches 73.82%.

The gap between training and validation accuracy indicates some degree of overfitting.

## Training and Validation Curves

The experiment plots training and validation accuracy over 10 epochs.

The training accuracy increases steadily, while validation accuracy improves more slowly and fluctuates later in training.

The corresponding loss curves show:

- Consistent decrease in training loss
- Initial decrease in validation loss
- Later fluctuations in validation loss

This behaviour is consistent with improving training performance and some degree of overfitting near the end of training.

## Comparison of CNN Architectures

The report includes the following architecture comparison:

| Model | Parameters | Accuracy (%) | Training Time |
|---|---:|---:|---:|
| AlexNet | 6,976,842 | 73.84 | 198.30 s |
| VGG16 | 14,848,586 | 73.82 | 200.00 s |
| ResNet50 | 24,114,826 | 78.56 | 365.00 s |
| GoogleNet | 2,206,346 | 65.00 | 90.00 s |
| LeNet-5 | 83,126 | 15.00 | 36.00 s |

These values are the comparison results reported in the experiment.

The supplied notebook implementation itself evaluates VGG16 transfer learning. The other architecture values are presented in the report's comparison table.

## Key Observations

- CNN architectures have evolved from relatively small networks such as LeNet-5 to much deeper architectures such as ResNet.
- AlexNet introduced important practical ideas including ReLU, dropout, GPU training and data augmentation.
- VGG16 uses repeated 3 x 3 convolutions and a deeper, uniform architecture.
- GoogleNet uses Inception modules for multi-scale feature extraction.
- ResNet uses residual connections to improve optimization in deep networks.
- Dilated convolution increases the receptive field without proportionally increasing kernel parameters.
- Transpose convolution is used for learnable upsampling.
- Transfer learning allows pretrained visual features to be reused for a new classification task.
- Fine tuning the last VGG16 convolutional block improved validation accuracy from 62.17% to 72.17%.
- The best hyperparameter experiment reported a test accuracy of 73.82%.
- The best model used Adam, a learning rate of 0.001, 256 dense units and partial layer freezing.
- The best model reached 87.52% training accuracy and 73.82% testing accuracy.
- The training and validation curves show some overfitting toward the later epochs.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
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
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

Run the notebook or Python implementation included in the repository.

## Project Structure

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

https://github.com/ssukesh182/DeeplearningLab
