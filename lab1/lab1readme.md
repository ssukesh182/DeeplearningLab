# Deep Learning Lab 1: Single Layer Perceptron

Implementation of a Single Layer Perceptron from scratch using Python for binary classification and logic gate operations.

## Overview

This experiment implements a Single Layer Perceptron without relying on a pre-built neural network implementation for the main model.

The project covers:

- Perceptron architecture and learning rule
- Binary classification using the Banknote Authentication dataset
- Exploratory Data Analysis
- Feature normalization using StandardScaler
- Perceptron implementation from scratch
- Model evaluation using Accuracy, Precision, Recall and F1 Score
- Comparison with Scikit-Learn's Perceptron
- Learning rate analysis
- Weight and bias evolution during training
- Perceptron implementation for AND, OR and NOT logic gates

## Dataset

The Banknote Authentication dataset is obtained from the UCI Machine Learning Repository.

The dataset contains 1,372 samples with four numerical features extracted from wavelet-transformed images of banknotes.

### Features

1. Variance
2. Skewness
3. Curtosis
4. Entropy

### Classes

- `0`: Genuine Note
- `1`: Forged Note

### Dataset Split

| Parameter | Value |
|---|---:|
| Total Samples | 1372 |
| Training Samples | 1097 |
| Testing Samples | 275 |
| Features | 4 |
| Classes | 2 |
| Missing Values | None |
| Normalization | StandardScaler |

## Perceptron

The perceptron calculates a weighted sum of the input features:

```text
z = w^T x + b
```

A step activation function is then applied:

```text
f(z) = 1, if z >= 0
       0, if z < 0
```

For every training sample, the prediction error is calculated as:

```text
e = y - y_hat
```

The weights and bias are updated using:

```text
w = w + eta * e * x
b = b + eta * e
```

Training continues for multiple epochs until the model converges or the maximum number of epochs is reached.

## Implementation

The experiment follows these steps:

1. Load the Banknote Authentication dataset.
2. Perform exploratory data analysis.
3. Check for missing values.
4. Normalize the features using StandardScaler.
5. Split the dataset into training and testing sets.
6. Initialize the perceptron weights and bias.
7. Train the model using the perceptron learning rule.
8. Track the evolution of weights and bias.
9. Evaluate the trained model.
10. Compare the implementation with Scikit-Learn.
11. Train the model using different learning rates.

## Results

The custom perceptron was trained for 100 epochs with a learning rate of `0.001`.

The best accuracy obtained during the learning rate analysis was with a learning rate of `0.0001`.

### Custom Perceptron

| Metric | Score |
|---|---:|
| Accuracy | 99.27% |
| Precision | 100.00% |
| Recall | 98.36% |
| F1 Score | 99.17% |

### Custom vs Scikit-Learn

| Metric | Custom Perceptron | Scikit-Learn |
|---|---:|---:|
| Accuracy | 99.27% | 100.00% |
| Precision | 100.00% | 100.00% |
| Recall | 98.36% | 100.00% |
| F1 Score | 99.17% | 100.00% |

## Learning Rate Analysis

The model was evaluated using multiple learning rates.

| Learning Rate | Accuracy |
|---:|---:|
| 0.0001 | 99.27% |
| 0.001 | 98.55% |
| 0.01 | 98.91% |
| 0.10 | 98.55% |
| 0.50 | 98.91% |
| 1.00 | 98.55% |

The experiments show that smaller learning rates resulted in smoother updates, while larger learning rates produced more fluctuations during training.

## Exploratory Data Analysis

The project includes the following visualizations:

- Class distribution
- Feature box plots
- Feature histograms
- Pair plot
- Correlation matrix
- Feature scatter plots

The pair plot and scatter plots show that several feature combinations provide noticeable separation between the two classes.

The correlation analysis also shows a strong negative correlation between Skewness and Curtosis.

## Weight and Bias Evolution

The implementation records the weight and bias values during training.

The weight plots show the parameters stabilizing as training progresses. The bias also approaches a stable value over successive epochs, indicating convergence.

## Logic Gate Implementation

The perceptron was also implemented for basic Boolean logic gates.

The model successfully learns the linearly separable:

- AND
- OR
- NOT

It cannot learn XOR because XOR is not linearly separable and therefore cannot be represented using a single linear decision boundary.

### Results

| Gate | Epochs to Converge | Total Updates | Final Weights | Final Bias |
|---|---:|---:|---|---:|
| AND | 5 | 11 | `[2.0, 1.0]` | `-3` |
| OR | 3 | 5 | `[1.0, 1.0]` | `-1` |
| NOT | 2 | 2 | `[-1.0]` | `0` |

The perceptron converges quickly for these gates because their decision boundaries are linear.

## Activation Function

The perceptron uses the Step activation function.

```text
f(x) = 1, if x >= 0
       0, if x < 0
```

Unlike sigmoid, tanh and ReLU, the step function is not differentiable and therefore is not suitable for gradient-based backpropagation used in deeper neural networks.

## Requirements

- Python 3.x
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
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the Python implementation using:

```bash
python <script_name>.py
```

Replace `<script_name>.py` with the appropriate Python file in the repository.

## Project Structure

The repository contains the implementation, dataset processing, model training and generated visualizations for the experiment.

A typical structure is:

```text
DeeplearningLab/
├── *.py
├── *.csv
├── plots/
└── README.md
```

The exact file names may vary depending on the implementation.

## Key Observations

- A single layer perceptron can solve linearly separable binary classification problems.
- Feature normalization provides more stable parameter updates.
- The Banknote Authentication dataset achieved 99.27% accuracy with the custom implementation.
- Scikit-Learn's implementation achieved 100% accuracy on the same experiment.
- Learning rate has a direct effect on the convergence behaviour of the perceptron.
- AND, OR and NOT can be represented using a single perceptron.
- XOR cannot be solved by a single layer perceptron because it requires a non-linear decision boundary.

## References

1. Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain", 1958.
2. Ian Goodfellow, Yoshua Bengio and Aaron Courville, *Deep Learning*, MIT Press.
3. Simon Haykin, *Neural Networks and Learning Machines*.
4. UCI Machine Learning Repository, Banknote Authentication Dataset.
5. Scikit-Learn Documentation.
