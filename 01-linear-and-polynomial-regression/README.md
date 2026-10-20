# 📈 Linear & Polynomial Regression Modeling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Topic](https://img.shields.io/badge/Topic-Supervised%20Regression-blue.svg)](#)

## 📌 Project Overview
This project explores linear supervised learning algorithms. It shows how linear equations model direct relationships and how polynomial feature transformations allow linear models to capture complex non-linear curves.

---

## 📚 Topics Covered
1. **Simple Linear Regression**:
   - Single input predictor (e.g., house size vs price).
   - OLS cost function minimization.
   - Calculating slope ($m$) and intercept ($c$): $y = mx + c$.
2. **Multiple Linear Regression**:
   - Multi-dimensional feature spaces: $\hat{y} = 	heta_0 + 	heta_1 x_1 + \dots + 	heta_n x_n$.
   - Matrix formulation and model fitting.
3. **Polynomial Regression**:
   - Transforming features using `PolynomialFeatures(degree=2)`.
   - Fitting non-linear curves to quadratic/exponential trends.
   - Comparative visualization of linear vs polynomial regression fits.

---

## 💻 Code Structure
- **Notebook**: [`linear_and_polynomial_regression.ipynb`](./linear_and_polynomial_regression.ipynb)
- **Cell Count**: 21 cells (18 code cells, 3 markdown cells)
- **Dependencies**: `scikit-learn`, `numpy`, `pandas`, `matplotlib`

---

## 🚀 How to Run
```bash
pip install scikit-learn numpy pandas matplotlib
jupyter notebook linear_and_polynomial_regression.ipynb
```
