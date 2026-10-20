# 🌲 Ensemble Learning: Bagging, Boosting & XGBoost

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Library-XGBoost-blue.svg)](https://xgboost.readthedocs.io/)
[![Significance](https://img.shields.io/badge/Significance-Top%20Tier%20Project-gold.svg)](#)

## 📌 Project Overview
Ensemble methods represent the state-of-the-art in classical machine learning competitions and production predictive systems. This project covers Bagging and Boosting architectures with extensive code implementations, hyperparameter curves, and head-to-head algorithm benchmarks.

---

## 🛠️ Experiments Conducted

### 1. Bagging with Decision Trees (Iris Classification)
- Constructing a `BaggingClassifier` with 50 decision tree base estimators.
- Accessing individual fitted base estimators (`estimators_[0]`).
- Extracting and plotting Gini feature importances across flower attributes.

### 2. Bagging with K-Nearest Neighbors
- Encapsulating non-tree base estimators (`KNeighborsClassifier`).
- Hyperparameter tuning across `n_estimators` from 1 to 50.
- Visualizing the stabilization curve of test accuracy as ensemble scale increases.

### 3. Boosting Benchmark: AdaBoost vs Gradient Boosting vs XGBoost
- Generating balanced multi-feature classification datasets.
- Performing stratified 70/30 train/test splitting and standardization.
- Training three state-of-the-art boosting implementations:
  1. `AdaBoostClassifier(n_estimators=50, learning_rate=1.0)`
  2. `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)`
  3. `XGBClassifier(n_estimators=100, learning_rate=0.1)`
- Comparative visualization of test accuracy across all three architectures.

---

## 💻 Code Structure
- **Notebook**: [`ensemble_learning_bagging_boosting.ipynb`](./ensemble_learning_bagging_boosting.ipynb)
- **Cell Count**: 36 cells (27 code cells, 9 markdown cells)
- **Total Code Lines**: 175 lines of machine learning code.
- **Dependencies**: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 🚀 How to Run
```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
jupyter notebook ensemble_learning_bagging_boosting.ipynb
```
