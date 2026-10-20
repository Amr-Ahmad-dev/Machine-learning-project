# ⚖️ Feature Engineering, Overfitting & Regularization (Ridge & Lasso)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Topic](https://img.shields.io/badge/Topic-Regularization%20%26%20Overfitting-red.svg)](#)

## 📌 Project Overview
Overfitting occurs when a model memorizes training noise rather than underlying patterns. This project demonstrates this critical concept through controlled simulation: preprocessing raw inputs, inducing severe high-variance overfitting with a 15th-degree polynomial, and resolving it with L1 (Lasso) and L2 (Ridge) penalties.

---

## 🛠️ Key Pipeline Stages

### 1. Categorical Feature Preprocessing
- **Label Encoding**: Mapping ordinal categories to integer labels using `LabelEncoder`.
- **One-Hot Encoding**: Converting nominal categories into binary indicator dummy columns using `OneHotEncoder`.

### 2. Feature Scaling
- **StandardScaler**: Standardizing distributions to zero mean and unit variance ($z = (x - \mu) / \sigma$).
- **MinMaxScaler**: Normalizing bounded ranges into $[0, 1]$.

### 3. Controlled Overfitting Simulation
- Synthetic quadratic data with elevated Gaussian noise.
- Transformed using high-order `PolynomialFeatures(degree=15)`.
- Evaluation of the catastrophic divergence between Training $R^2$ and Testing $R^2$.

### 4. Mathematical Regularization Techniques
- **Lasso Regression (L1 Penalty)**:
  $$\min_w \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \|w\|_1$$
  Drives uninformative feature weights to absolute zero, acting as automatic feature selection.
- **Ridge Regression (L2 Penalty)**:
  $$\min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2$$
  Shrinks regression coefficients asymptotically towards zero without removing features.

---

## 💻 Code Structure
- **Notebook**: [`feature_engineering_and_regularization.ipynb`](./feature_engineering_and_regularization.ipynb)
- **Cell Count**: 27 cells (16 code cells, 11 markdown cells)
- **Dependencies**: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scipy`

---

## 🚀 How to Run
```bash
pip install scikit-learn numpy pandas matplotlib scipy
jupyter notebook feature_engineering_and_regularization.ipynb
```
