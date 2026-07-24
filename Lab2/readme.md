# Deep Learning Laboratory – Experiment 2

# Multi-Layer Perceptron (MLP) for Fashion-MNIST Image Classification

## Overview

This experiment implements a **Multi-Layer Perceptron (MLP)** using TensorFlow/Keras for multi-class image classification on the Fashion-MNIST dataset.

A baseline neural network is first developed and evaluated. Automated hyperparameter optimization using **Randomized Search with Cross Validation** is then performed to identify a computationally efficient model configuration.

The performance of both models is compared using multiple evaluation metrics and visualizations.

---

## Objectives

- Build an MLP using TensorFlow/Keras.
- Train the network on the Fashion-MNIST dataset.
- Perform automated hyperparameter optimization.
- Compare baseline and optimized models.
- Evaluate classification performance using multiple metrics.

---

## Dataset

**Dataset:** Fashion-MNIST

| Property | Value |
|----------|-------|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 |
| Channels | 1 |
| Input Features | 784 |
| Number of Classes | 10 |
| Classification Type | Multi-Class |

### Classes

| Label | Class |
|------:|-------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## Data Preprocessing

- Pixel normalization
- Flattening images from 28×28 to 784 features
- Dataset visualization
- Class distribution analysis

---

## Baseline Model Architecture

| Layer | Configuration |
|--------|--------------|
| Input | 784 |
| Hidden Layer 1 | 128 Neurons (ReLU) |
| Hidden Layer 2 | 64 Neurons (ReLU) |
| Output | 10 Neurons (Softmax) |

---

## Training Configuration

| Parameter | Value |
|-----------|------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 20 |
| Batch Size | 32 |
| Training Time | 179.96 s |

---

## Baseline Performance

| Metric | Value |
|---------|------:|
| Accuracy | **88.85%** |
| Precision | **88.93%** |
| Recall | **88.85%** |
| F1 Score | **88.76%** |

---

## Hyperparameter Optimization

Randomized Search with Cross Validation was performed over the following search space.

| Hyperparameter | Candidate Values |
|---------------|-----------------|
| Hidden Layers | 1, 2, 3 |
| Hidden Neurons | 32, 64, 128, 256 |
| Activation | ReLU, Sigmoid, Tanh |
| Optimizer | SGD, Adam, RMSProp |
| Learning Rate | 0.1, 0.01, 0.001 |
| Batch Size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |
| Dropout | 0.0, 0.2, 0.5 |

---

## Best Hyperparameters

| Parameter | Best Value |
|-----------|-----------|
| Hidden Layers | 1 |
| Hidden Neurons | 256 |
| Activation | Sigmoid |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 10 |
| Dropout | 0.0 |
| Cross Validation Accuracy | **88.11%** |

---

## Optimized Model Performance

| Metric | Value |
|---------|------:|
| Accuracy | **87.67%** |
| Precision | **87.67%** |
| Recall | **87.67%** |
| F1 Score | **87.56%** |
| Training Time | **51.80 s** |

---

## Performance Comparison

| Metric | Baseline | Optimized |
|---------|---------:|----------:|
| Accuracy | **88.85%** | **87.67%** |
| Precision | **88.93%** | **87.67%** |
| Recall | **88.85%** | **87.67%** |
| F1 Score | **88.76%** | **87.56%** |
| Training Time | **179.96 s** | **51.80 s** |

---

## Generated Visualizations

The notebook generates the following plots.

- Sample Fashion-MNIST Images
- Dataset Class Distribution
- Baseline Confusion Matrix
- Baseline Accuracy vs Epoch
- Baseline Loss vs Epoch
- Optimized Confusion Matrix
- Optimized Accuracy vs Epoch
- Optimized Loss vs Epoch
- Hyperparameter Search Results
- Model Accuracy Comparison

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Repository Structure

```
Experiment_2/
│
├── deeplearninglab2.ipynb
├── README.md
├── MLP_Plots/
│   ├── *.pdf
│   └── *.png
└── Report/
    └── Experiment_2_Report.pdf
```

---

## Conclusion

The baseline MLP achieved an accuracy of **88.85%** on the Fashion-MNIST dataset.

Randomized Search successfully identified a simpler neural network architecture that reduced training time from **179.96 seconds** to **51.80 seconds** while maintaining competitive classification performance.

The experiment demonstrates the importance of hyperparameter optimization in improving computational efficiency while preserving model accuracy.

---

## References

- TensorFlow Documentation
- Keras Documentation
- Fashion-MNIST Dataset
- Scikit-learn Documentation
- Deep Learning by Ian Goodfellow