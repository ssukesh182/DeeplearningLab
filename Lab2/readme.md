# Deep Learning Laboratory – Experiment 1

## Single Layer Perceptron for Binary Classification

### Overview

This experiment demonstrates the implementation of a **Single Layer Perceptron** from scratch using Python for binary classification. The performance of the custom implementation is compared with Scikit-learn's built-in Perceptron classifier using the Banknote Authentication dataset.

The experiment also studies the effect of different learning rates on the convergence and classification performance of the perceptron.

---

## Objectives

- Implement a Single Layer Perceptron from scratch.
- Train the model using the Perceptron Learning Algorithm.
- Evaluate model performance using standard classification metrics.
- Compare the custom implementation with Scikit-learn.
- Analyze the influence of different learning rates.

---

## Dataset

**Dataset:** Banknote Authentication Dataset

| Property | Value |
|----------|-------|
| Total Samples | 1372 |
| Training Samples | 1097 |
| Testing Samples | 275 |
| Number of Features | 4 |
| Classes | 2 |
| Task | Binary Classification |

### Features

- Variance
- Skewness
- Curtosis
- Entropy

---

## Data Preprocessing

- Missing value verification
- Feature standardization using StandardScaler
- Train-Test Split
- Feature normalization

---

## Model Implementation

### Custom Perceptron

- Learning Algorithm
- Weight Initialization
- Bias Update
- Binary Step Activation
- Iterative Weight Optimization

### Scikit-learn Perceptron

Used as the baseline implementation for comparison.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Performance Metrics

### Custom Perceptron

| Metric | Value |
|---------|------:|
| Accuracy | **99.27%** |
| Precision | **100.00%** |
| Recall | **98.36%** |
| F1 Score | **99.17%** |

---

### Scikit-learn Perceptron

| Metric | Value |
|---------|------:|
| Accuracy | **100.00%** |

---

## Learning Rate Analysis

| Learning Rate | Accuracy |
|--------------:|---------:|
| 0.0001 | 99.27% |
| 0.001 | 98.55% |
| 0.01 | 98.91% |
| 0.10 | 98.55% |
| 0.50 | 98.91% |
| 1.00 | 98.55% |

---

## Experimental Results

The custom implementation achieved nearly perfect classification accuracy while closely matching the performance of Scikit-learn's implementation.

The model converged rapidly for lower learning rates, whereas larger learning rates introduced minor fluctuations in classification accuracy.

---

## Generated Visualizations

The notebook generates the following plots:

- Dataset Distribution
- Feature Correlation Heatmap
- Decision Boundary
- Confusion Matrix
- Learning Rate Comparison
- Accuracy Comparison
- Weight Evolution
- Error Curve
- Classification Report Visualization
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Prediction Distribution
- Training Analysis
- Additional Experimental Graphs

---

## Repository Structure

```
Experiment_1/
│
├── deeplearninglab1.ipynb
├── README.md
├── Plots/
│   ├── *.pdf
│   └── *.png
└── Report/
    └── Experiment_1_Report.pdf
```

---

## Conclusion

The Single Layer Perceptron successfully classified the Banknote Authentication dataset with an accuracy exceeding 99%. The experiment demonstrates that even a simple linear classifier can perform exceptionally well on linearly separable datasets.

Comparison with Scikit-learn validates the correctness of the custom implementation.

---

## References

- TensorFlow Documentation
- Scikit-learn Documentation
- UCI Machine Learning Repository
- Deep Learning by Ian Goodfellow
