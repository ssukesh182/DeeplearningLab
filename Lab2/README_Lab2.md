# Deep Learning Lab 2: Multi-Layer Perceptron

Implementation of a Multi-Layer Perceptron (MLP) using TensorFlow/Keras for multi-class image classification on the Fashion-MNIST dataset.

The experiment also includes automated hyperparameter optimization using Randomized Search and a comparison between the baseline and optimized models.

## Overview

This experiment covers:

- Multi-Layer Perceptron architecture
- Multi-class image classification
- Fashion-MNIST dataset preprocessing
- ReLU and Softmax activation functions
- Sparse Categorical Cross Entropy loss
- Adam optimization
- Model training and evaluation
- Confusion matrix analysis
- Accuracy, Precision, Recall and F1 Score
- Automated hyperparameter optimization using Randomized Search
- Five-fold cross validation
- Comparison of baseline and optimized models
- XOR classification using a single-layer perceptron
- XOR classification using an MLP

## Dataset

The experiment uses the Fashion-MNIST dataset.

Fashion-MNIST contains grayscale images of clothing items belonging to 10 classes.

| Parameter | Value |
|---|---:|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 x 28 |
| Image Type | Grayscale |
| Number of Classes | 10 |
| Input Dimension | 784 |
| Task | Multi-Class Classification |

### Classes

1. T-shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

## Data Preprocessing

The pixel values are normalized to the range `[0, 1]` by dividing each pixel value by 255.

Each `28 x 28` image is then flattened into a vector containing 784 features.

```text
28 x 28 image
      |
      v
  Flatten
      |
      v
784 input features
```

## Baseline MLP Architecture

The baseline model consists of two hidden layers.

```text
Input
784 neurons
    |
    v
Dense Layer
128 neurons
ReLU
    |
    v
Dense Layer
64 neurons
ReLU
    |
    v
Output Layer
10 neurons
Softmax
```

### Baseline Configuration

| Parameter | Value |
|---|---|
| Input Size | 784 |
| Hidden Layer 1 | 128 neurons, ReLU |
| Hidden Layer 2 | 64 neurons, ReLU |
| Output Layer | 10 neurons, Softmax |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | Sparse Categorical Cross Entropy |
| Batch Size | 32 |
| Epochs | 20 |
| Training Time | 179.96 seconds |

## Activation Functions

### ReLU

The hidden layers use the Rectified Linear Unit:

```text
f(x) = max(0, x)
```

ReLU is computationally inexpensive and helps reduce the vanishing gradient problem.

### Softmax

The output layer uses Softmax to convert the output scores into class probabilities.

```text
P(y = i) = exp(z_i) / sum(exp(z_j))
```

The probabilities across all 10 classes sum to one.

## Loss Function

The models use Sparse Categorical Cross Entropy.

For a training sample:

```text
L = -log(P_correct)
```

where `P_correct` is the probability assigned to the correct class.

## Optimizer

The baseline and optimized models use the Adam optimizer.

Adam provides adaptive learning rates and combines ideas from momentum-based optimization with adaptive parameter updates.

## Training Procedure

The experiment follows these steps:

1. Import the required Python libraries.
2. Load the Fashion-MNIST dataset using TensorFlow.
3. Explore sample images and class distribution.
4. Normalize the image pixel values.
5. Flatten the images into 784-dimensional vectors.
6. Construct the baseline MLP.
7. Compile the model using Adam and Sparse Categorical Cross Entropy.
8. Train the baseline model for 20 epochs with a batch size of 32.
9. Evaluate the model on the test dataset.
10. Calculate Accuracy, Precision, Recall and F1 Score.
11. Generate the confusion matrix and training plots.
12. Perform Randomized Search with five-fold cross validation.
13. Train the model using the selected hyperparameters.
14. Compare the baseline and optimized models.

## Hyperparameter Optimization

Randomized Search was used to explore different combinations of model and training parameters.

### Search Space

| Hyperparameter | Candidate Values |
|---|---|
| Hidden Layers | 1, 2, 3 |
| Hidden Neurons | 32, 64, 128, 256 |
| Activation Function | ReLU, Tanh, Sigmoid |
| Optimizer | SGD, Adam, RMSProp |
| Learning Rate | 0.1, 0.01, 0.001 |
| Batch Size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |
| Dropout Rate | 0.0, 0.2, 0.5 |

### Best Hyperparameters

The Randomized Search selected the following configuration:

| Parameter | Selected Value |
|---|---|
| Hidden Layers | 1 |
| Hidden Neurons | 256 |
| Activation Function | Sigmoid |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 10 |
| Dropout | 0.0 |
| Cross Validation Accuracy | 88.11% |

## Results

### Baseline Model

| Metric | Value |
|---|---:|
| Accuracy | 88.85% |
| Precision | 88.93% |
| Recall | 88.85% |
| F1 Score | 88.76% |
| Training Time | 179.96 s |

### Optimized Model

| Metric | Value |
|---|---:|
| Accuracy | 87.67% |
| Precision | 87.67% |
| Recall | 87.67% |
| F1 Score | 87.56% |
| Training Time | 51.80 s |

### Baseline vs Optimized

| Metric | Baseline | Optimized |
|---|---:|---:|
| Accuracy | 88.85% | 87.67% |
| Precision | 88.93% | 87.67% |
| Recall | 88.85% | 87.67% |
| F1 Score | 88.76% | 87.56% |
| Training Time | 179.96 s | 51.80 s |
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.001 |
| Batch Size | 32 | 64 |
| Activation | ReLU | Sigmoid |
| Hidden Layers | 2 | 1 |

The optimized model has slightly lower test performance than the baseline, but its training time is substantially lower. The training time decreased from 179.96 seconds to 51.80 seconds.

## Result Analysis

The experiment includes the following visualizations:

- Sample Fashion-MNIST images
- Class distribution
- Baseline confusion matrix
- Baseline training and validation accuracy
- Baseline training and validation loss
- Optimized model confusion matrix
- Optimized training and validation accuracy
- Optimized training and validation loss
- Hyperparameter search results
- Baseline vs optimized model accuracy

### Confusion Matrix

The baseline confusion matrix shows that most predictions fall along the diagonal.

The main classification errors occur between visually similar categories such as:

- Shirt
- Pullover
- Coat

This is expected because these classes have similar visual characteristics.

The optimized model also shows a strong diagonal pattern, although its overall test accuracy is slightly lower.

### Training Curves

For the baseline model, training accuracy increases over the epochs while the training loss decreases.

For the optimized model, the loss decreases rapidly during the initial epochs before stabilizing.

## XOR: Perceptron Limitation and MLP Solution

The experiment also demonstrates why a single-layer perceptron cannot solve the XOR problem and how an MLP overcomes this limitation.

### XOR Truth Table

| x1 | x2 | Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR is not linearly separable. No single straight-line decision boundary can correctly separate the two classes.

### Single-Layer Perceptron

A single-layer perceptron was trained using:

- Weights initialized to zero
- Bias initialized to zero
- Step activation
- Learning rate = 1
- Maximum of 10 epochs

The perceptron does not converge on XOR.

Instead, the weights enter a repeating update cycle. After 10 epochs, the model still misclassifies the XOR inputs.

## MLP for XOR

An MLP with the following architecture was used:

```text
Input Layer
2 neurons
    |
    v
Hidden Layer
4 neurons
Sigmoid
    |
    v
Output Layer
1 neuron
Sigmoid
```

The network was trained using backpropagation for 10,000 epochs with:

```text
Learning Rate = 0.1
Loss = Mean Squared Error
```

### XOR Training Loss

| Epoch | MSE Loss |
|---:|---:|
| 1000 | 0.2389 |
| 2000 | 0.1855 |
| 3000 | 0.0945 |
| 4000 | 0.0263 |
| 5000 | 0.0116 |
| 6000 | 0.0069 |
| 7000 | 0.0047 |
| 8000 | 0.0036 |
| 9000 | 0.0028 |
| 10000 | 0.0023 |

### XOR Predictions

| x1 | x2 | Expected | Raw Prediction | Binary Prediction |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.0524 | 0 |
| 0 | 1 | 1 | 0.9536 | 1 |
| 1 | 0 | 1 | 0.9537 | 1 |
| 1 | 1 | 0 | 0.0476 | 0 |

The MLP correctly learns all four XOR combinations.

## Key Observations

- An MLP can learn non-linear relationships that a single-layer perceptron cannot.
- The Fashion-MNIST dataset can be classified using a fully connected neural network after flattening the images.
- The baseline MLP achieved 88.85% test accuracy.
- Randomized Search selected a simpler one-hidden-layer architecture.
- The optimized model achieved 87.67% test accuracy.
- The optimized model reduced training time from 179.96 seconds to 51.80 seconds.
- The baseline model produced better test metrics in this experiment, while the optimized model required substantially less training time.
- Visually similar Fashion-MNIST classes account for a significant portion of the classification errors.
- A single-layer perceptron cannot solve XOR because XOR is not linearly separable.
- Adding a hidden layer and non-linear activation allows an MLP to learn XOR.
- The XOR MLP reached an MSE of 0.0023 after 10,000 epochs and correctly classified all four input combinations.

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

The repository contains the implementation, dataset processing, model training, hyperparameter search and generated visualizations for the experiment.

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

1. Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning*, MIT Press.
2. TensorFlow Documentation
3. Keras Documentation
4. Fashion-MNIST Dataset
5. Scikit-Learn Documentation
