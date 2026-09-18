# Deep Learning Lab 5: CNN Training, Regularization, Optimization and Transfer Learning

A comparative study of CNN training strategies using MobileNetV2 and the Oxford-IIIT Pet dataset.

This experiment studies the effect of weight initialization, regularization, optimization algorithms, CNN hyperparameters, transfer learning, fine tuning and cross-validation on image classification performance.

## Overview

The experiment covers:

- Weight initialization
- Overfitting and regularization
- L2 regularization
- Batch Normalization
- Dropout
- SGD, Momentum, RMSProp and Adam
- CNN hyperparameter tuning
- Transfer learning using MobileNetV2
- Feature extraction
- Fine tuning
- 5-fold cross-validation
- Final model evaluation
- Accuracy, precision, recall and F1-score
- Training time and parameter comparison

## Dataset

The experiment uses the Oxford-IIIT Pet Dataset.

The dataset contains images of cats and dogs belonging to 37 breeds.

| Parameter | Value |
|---|---:|
| Number of Classes | 37 |
| Image Type | RGB |
| Input Size | 224 x 224 x 3 |
| Base Model | MobileNetV2 |
| Pretrained Weights | ImageNet |

The images have different spatial dimensions originally and are resized to `224 x 224 x 3`.

Input images are normalized according to the preprocessing requirements of MobileNetV2.

The experiment maintains separate training, validation and test data. The test set remains untouched during hyperparameter selection.

## Why MobileNetV2

MobileNetV2 is used as the base model because it is a lightweight CNN architecture designed to reduce computational cost while maintaining useful image representations.

It is particularly suitable for CPU-based laboratory work.

Important components of MobileNetV2 include:

- Depthwise convolution
- Pointwise 1 x 1 convolution
- Inverted residual blocks
- Linear bottlenecks
- Batch Normalization
- ReLU6 activation

A simplified representation is:

```text
Input
224 x 224 x 3
    |
    v
Convolution
    |
    v
Depthwise Convolution
    |
    v
Pointwise Convolution
    |
    v
Residual Block
    |
    v
Global Average Pooling
    |
    v
Dense
    |
    v
Softmax
```

## Weight Initialization

Weight initialization determines the initial values of trainable weights before training.

In this experiment, the MobileNetV2 feature extractor is already pretrained, so initialization mainly affects the newly added classification layer.

Different initialization methods are compared using:

- Training loss vs epoch
- Validation accuracy vs epoch

The loss curves show different convergence behaviour depending on the initialization method. The validation accuracy curves show how the starting weights influence the early training behaviour of the classifier.

## Regularization and Overfitting

Overfitting occurs when a model performs well on training data but does not generalize as well to unseen data.

The experiment compares training and validation behaviour under different regularization settings.

### No Regularization

Without regularization, training accuracy increases strongly while the validation behaviour improves more slowly.

The training loss also decreases significantly, creating a larger difference between training and validation behaviour.

### L2 Regularization

L2 regularization is used to penalize large weights.

The experiment shows a reduction in both training and validation loss and a smaller difference between the two curves.

This indicates improved generalization for the tested configuration.

## Batch Normalization

Batch Normalization normalizes activations within a mini-batch during training.

For a mini-batch:

```text
x1, x2, ..., xm
```

the batch mean is:

```text
mu_B = (1/m) * sum(x_i)
```

and the variance is:

```text
sigma_B^2 = (1/m) * sum((x_i - mu_B)^2)
```

The normalized activation is:

```text
x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon)
```

The final output is:

```text
y_i = gamma * x_hat_i + beta
```

where `gamma` and `beta` are learnable parameters.

### Numerical Example

For:

```text
x = [2, 4, 6, 8]
```

the experiment gives:

```text
Mean = 5
Variance = 5
```

Ignoring the small epsilon:

```text
x_hat = [-1.342, -0.447, 0.447, 1.342]
```

With:

```text
gamma = 1
beta = 0
```

the normalized values are also the outputs.

The Batch Normalization experiment shows more stable validation behaviour during training.

## Optimization Algorithms

The experiment compares four optimization algorithms:

- SGD
- Momentum
- RMSProp
- Adam

The training loss and validation accuracy curves are used to compare their convergence behaviour.

### Optimizer Results

| Optimizer | Final Loss | Best Validation Accuracy | Training Time |
|---|---:|---:|---:|
| SGD | 2.7655 | 29.40% | 636.09 s |
| Momentum | 0.5602 | 85.00% | 653.26 s |
| RMSProp | 0.0849 | 89.80% | 588.52 s |
| Adam | 0.1082 | 88.00% | 652.53 s |

The experiment shows faster loss reduction for Momentum, RMSProp and Adam compared with standard SGD.

## CNN Hyperparameter Tuning

The following hyperparameters are investigated:

| Hyperparameter | Values Investigated |
|---|---|
| Learning Rate | 0.001, 0.0001 |
| Batch Size | 16, 32, 64 |
| Dropout Rate | 0, 0.25, 0.5 |
| Optimizer | SGD, Momentum, RMSProp, Adam |
| Fine-Tuning Learning Rate | 10^-5 |

For a convolution operation, the output dimension is calculated using:

```text
O = floor((N + 2P - K) / S) + 1
```

where:

- `N` = input size
- `K` = kernel size
- `P` = padding
- `S` = stride

During the controlled hyperparameter study, one hyperparameter is changed at a time while the remaining settings are kept fixed.

### Learning Rate

| Learning Rate | Best Validation Accuracy |
|---:|---:|
| 0.001 | 89.8% |
| 0.0001 | 80.4% |

For this experiment, the learning rate of `0.001` produced higher validation accuracy.

### Batch Size

Batch sizes of `16`, `32` and `64` were compared.

The validation accuracy was similar across the three configurations, with batch size `32` giving the highest result at approximately `89.2%`.

### Dropout

Dropout rates of `0`, `0.25` and `0.5` were compared.

A dropout rate of `0.5` produced the highest validation accuracy in this experiment, although the difference between the configurations was small.

The dropout training curves show improved training behaviour while validation remains relatively stable.

## Transfer Learning

MobileNetV2 pretrained on ImageNet is used for transfer learning.

Two approaches are compared.

### Case A: Feature Extraction

```text
Pretrained MobileNetV2
        |
     Freeze Base
        |
   New Classifier
```

The pretrained convolutional layers are frozen and only the newly added classification layers are trained.

### Case B: Fine Tuning

```text
Pretrained MobileNetV2
        |
Unfreeze Selected Layers
        |
Small Learning Rate
        |
Fine Tune
```

The upper layers of the pretrained network are unfrozen and trained using a smaller learning rate.

Fine tuning allows the pretrained feature extractor to adapt to the target dataset.

A smaller learning rate is important during fine tuning because large updates can damage useful pretrained features, a problem commonly referred to as catastrophic forgetting.

## Feature Extraction vs Fine Tuning

The experiment compares validation accuracy and validation loss for feature extraction and fine tuning.

Fine tuning reaches a slightly higher peak validation accuracy in the reported experiment.

The improvement is not dramatic because ImageNet pretrained features are already useful for the target image classification task.

Fine tuning also requires additional computation and more careful optimization.

## 5-Fold Cross-Validation

Three configurations are evaluated using 5-fold cross-validation.

### Configurations

```text
C1:
Dropout = 0.0
Optimizer = Adam
Learning Rate = 0.001

C2:
Dropout = 0.25
Optimizer = Adam
Learning Rate = 0.001

C3:
Dropout = 0.5
Optimizer = Adam
Learning Rate = 0.0001
```

For five folds, the mean accuracy is calculated as:

```text
A_mean = (1/5) * sum(A_i)
```

The standard deviation is calculated from the variation between the five fold accuracies.

### Cross-Validation Results

| Configuration | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 85.0 | 85.0 | 87.0 | 88.5 | 84.0 | 85.9 ± 1.62 |
| C2 | 86.5 | 83.5 | 87.5 | 84.5 | 80.0 | 84.4 ± 2.62 |
| C3 | 26.0 | 8.0 | 16.0 | 14.5 | 17.0 | 16.3 ± 5.78 |

C1 has a mean cross-validation accuracy of `85.9%` with a standard deviation of `1.62%`.

C3 performs poorly in the short training schedule because the combination of high dropout and a very small learning rate does not learn sufficiently.

## Final Model

The configuration selected for final evaluation is:

```text
Dropout = 0.0
Optimizer = Adam
Learning Rate = 0.001
```

This configuration is retrained and evaluated on the untouched test set.

### Final Evaluation

| Metric | Value |
|---|---:|
| Mean CV Accuracy | 85.90% |
| CV Standard Deviation | 1.62% |
| Test Accuracy | 89.00% |
| Precision | 90.22% |
| Recall | 89.00% |
| F1 Score | 88.81% |
| Training Time | 670.86 s |
| Number of Parameters | 2,305,381 |

The confusion matrix shows a strong diagonal pattern, indicating that the model generally distinguishes the pet-breed classes correctly. Off-diagonal values represent classification errors between visually similar breeds.

## Overall Results

The report compares the main configurations using cross-validation accuracy, variability, test accuracy and training time.

| Configuration | CV Accuracy | SD | Test Accuracy | Training Time |
|---|---:|---:|---:|---:|
| Baseline | 85.90% | 1.62% | 89.00% | 670.86 s |
| Best Initialization | 88.70% | 1.40% | 90.00% | 706.82 s |
| Best Regularization | 85.90% | 1.62% | 89.00% | 670.86 s |
| Best Optimizer | 85.20% | 2.66% | 88.20% | 637.86 s |
| Best Hyperparameters | 85.90% | 1.62% | 89.00% | 670.86 s |
| Fine-Tuned Model | 80.60% | 2.06% | 89.60% | 1769.49 s |

The report emphasizes that model selection should consider more than a single accuracy value. Cross-validation variability and computational cost are also considered when evaluating configurations.

## Key Observations

- Weight initialization affects the convergence behaviour of the newly added classifier layers.
- Regularization can reduce the gap between training and validation performance.
- L2 regularization reduced both training and validation loss in the reported experiment.
- Batch Normalization produced more stable validation behaviour.
- SGD converged more slowly than the other tested optimizers.
- RMSProp achieved 89.80% best validation accuracy in the optimizer experiment.
- A learning rate of 0.001 performed better than 0.0001 in the tested configuration.
- Batch size had relatively little effect within the tested range of 16, 32 and 64.
- Dropout rate 0.5 produced the highest validation accuracy in its controlled comparison, although the difference was small.
- Transfer learning allows a pretrained MobileNetV2 feature extractor to be reused for the Oxford-IIIT Pet classification task.
- Fine tuning can improve adaptation, but requires more computation and careful optimization.
- C1 achieved 85.90% mean cross-validation accuracy with 1.62% standard deviation.
- The final C1 model achieved 89.00% test accuracy, 90.22% precision and 88.81% F1-score.
- The best initialization configuration in the overall results table achieved 90.00% test accuracy, while C1 was selected based on the reported balance of validation performance, variability and computational cost.
- The fine-tuned model required substantially more training time than the regular configuration.

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
git clone https://github.com/ssukesh/DeeplearningLab.git
cd DeeplearningLab
```

Install the required packages:

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
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
5. Oxford-IIIT Pet Dataset Documentation.

## Source Code

https://github.com/ssukesh/DeeplearningLab
