# 🎯 Supervised Classification: KNN & Support Vector Machines (SVM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Topic](https://img.shields.io/badge/Topic-Supervised%20Classification-purple.svg)](#)

## 📌 Project Overview
This project presents an empirical and visual study of two fundamental supervised classification algorithms: K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). It emphasizes hyperparameter tuning, cross-validation, and decision surface visualization.

---

## 📚 Topics Covered
1. **Synthetic Dataset Generation**:
   - Creating non-trivial 2D classification problems with `make_classification`.
   - Stratified splitting into training (70%) and testing (30%) subsets.
2. **K-Nearest Neighbors (KNN)**:
   - Principle of distance metrics (Euclidean) and majority voting.
   - Hyperparameter search over $k \in [1, 10]$ via `GridSearchCV(cv=5)`.
   - Plotting the $k$ versus validation accuracy curve to isolate the optimal neighborhood size.
3. **Support Vector Machines (SVM)**:
   - Objective: Maximizing the geometric margin $2 / \|w\|$.
   - Fitting `SVC(kernel='linear', C=1.0)`.
   - Generating 2D meshgrids to contour the decision boundaries and margins.

---

## 💻 Code Structure
- **Notebook**: [`knn_and_svm_classification.ipynb`](./knn_and_svm_classification.ipynb)
- **Cell Count**: 22 cells (20 code cells, 2 markdown cells)
- **Dependencies**: `scikit-learn`, `numpy`, `matplotlib`

---

## 🚀 How to Run
```bash
pip install scikit-learn numpy matplotlib
jupyter notebook knn_and_svm_classification.ipynb
```
