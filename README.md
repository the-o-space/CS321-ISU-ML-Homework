# CS321 Machine Learning - Homework Solutions

This repository contains solutions for the CS321 Machine Learning course homework assignments.

## Project Structure

```
CS321 ISU/
├── .venv/              # Python virtual environment
├── requirements.txt    # Python dependencies
├── README.md          # This file
│
├── Homework 01/       # Matrix operations and optimization
│   ├── Homework_01.ipynb           # Original assignment
│   ├── Homework_01_Solution.ipynb  # Solution notebook
│   └── image.png                   # Reference image
│
├── Homework 02/       # Data analysis with Pandas and Linear Regression
│   ├── Homework_02.ipynb           # Original assignment
│   └── Homework_02_Solution.ipynb  # Solution notebook
│
├── Homework 03/       # Neural Networks
│   ├── Homework_03.ipynb           # Original assignment
│   ├── Homework_03_Solution.ipynb  # Solution notebook
│   ├── image.png                   # Reference images
│   └── image 2.png
│
├── Homework 04/       # SVM and Clustering
│   ├── Homework_04.ipynb           # Original assignment
│   ├── Homework_04_Solution.ipynb  # Solution notebook
│   └── image.png                   # Reference image
│
└── Homework 05/       # Statistical Learning and Dimensionality Reduction
    ├── Homework_05.ipynb           # Original assignment
    ├── Homework_05_Solution.ipynb  # Solution notebook
    ├── image.png                   # Reference images
    └── image 02.png
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd CS321\ ISU
```

### 2. Create and activate virtual environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

## Homework Assignments Overview

### Homework 01: Matrix Operations and Optimization
- **Problem 1**: Matrix Norms (1-norm, ∞-norm, sum-norm, Frobenius norm, nuclear norm)
- **Problem 2**: Vandermonde Matrix (determinant derivation and ill-conditioning)
- **Problem 3**: Multi-Class Confusion Matrix (metrics calculation)
- **Problem 4**: Finding Extrema (gradient, Hessian, critical points)
- **Extra Problem**: Gradient Descent Method

### Homework 02: Data Analysis and Machine Learning
- **Problem 1**: Data Analysis with Pandas (tips dataset)
- **Problem 2**: Visualization with Seaborn
- **Problem 3**: Data Preprocessing (categorical to numerical)
- **Problem 4**: Building Linear Regression Model
- **Extra Problem**: Convex Hull and Linear Separability

### Homework 03: Neural Networks
- **Problem 1**: Building a Single Neuron
- **Problem 2**: Combining Neurons into a Neural Network
- **Problem 3**: Loss Function Calculation (MSE)
- **Problem 4**: Backpropagation and Partial Derivatives
- **Extra Problem**: Training a Neural Network with SGD

### Homework 04: Support Vector Machines and Clustering
- **Problem 1**: Kernel Validation
- **Problem 2**: Maximum Margin Classifier
- **Problem 3**: SVM Classifiers on Iris Dataset
- **Problem 4**: Hierarchical Clustering
- **Extra Problem 1**: K-Means Clustering
- **Extra Problem 2**: Color Quantization

### Homework 05: Statistical Learning and Dimensionality Reduction
- **Problem 1**: Bias of an Estimator
- **Problem 2**: SVD Computation
- **Problem 3**: PCA Image Compression
- **Problem 4**: Kullback-Leibler Divergence
- **Extra Problem**: t-SNE on Donut Dataset

## Key Libraries Used

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Scientific computing

## Notes

- Each solution notebook contains detailed explanations and step-by-step implementations
- Mathematical derivations are included where required
- Visualizations are provided to help understand the concepts
- Code is documented with comments for clarity

## Requirements

- Python 3.8+
- Jupyter Notebook
- All packages listed in `requirements.txt`

## Author

Fedor Ryzhenkov

## License

This project is for educational purposes as part of the CS321 Machine Learning course. 