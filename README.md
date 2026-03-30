# Linear Regression Comparison

A comparative study of four approaches to linear regression in Python,
ranging from manual gradient descent to high-level statistical libraries.
Each implementation is explored on the same dataset so results are directly
comparable across methods.

---

## Methods

| # | Method | Library | Level |
|---|--------|---------|-------|
| 1 | Gradient Descent (manual) | NumPy | Low-level |
| 2 | Normal Equation (matrix multiplication) | NumPy | Low-level |
| 3 | Ordinary Least Squares | scikit-learn | High-level |
| 4 | Ordinary Least Squares | statsmodels | High-level + stats |

---



## Methods Overview

### 1. Gradient Descent (Manual)

Implements the update rule from scratch using NumPy:

$$\theta := \theta - \alpha \cdot \frac{1}{m} X^T (X\theta - y)$$

Parameters like learning rate (`α`) and number of iterations are tuned
manually. This method illustrates how the optimizer converges to the
solution step by step, and makes the cost function trajectory visible.

### 2. Normal Equation (Matrix Multiplication)

Computes the closed-form analytical solution directly:

$$\theta = (X^T X)^{-1} X^T y$$

No iteration is required. This method is exact (up to floating-point
precision) and serves as the numerical ground truth for comparing the
convergence of gradient descent.

### 3. scikit-learn — `LinearRegression`

Uses `sklearn.linear_model.LinearRegression`, which internally uses
the same least-squares solution via SVD decomposition. Provides a clean,
production-ready API and integrates naturally with pipelines, cross-
validation, and preprocessing utilities.

### 4. statsmodels — `OLS`

Uses `statsmodels.formula.api.ols` or `statsmodels.api.OLS`. Unlike
scikit-learn, statsmodels is oriented toward statistical inference — it
provides p-values, confidence intervals, R², adjusted R², F-statistics,
and a full model summary. Useful when the goal is understanding the model,
not just prediction.

---

## Metrics Compared

All four models are evaluated on the same train/test split:

- **MSE** — Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination
- **Convergence** — (Gradient Descent only) cost curve over iterations

---

## Results Summary

> Results are filled in after running all notebooks.

| Method | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Gradient Descent | | | | |
| Normal Equation | | | | |
| scikit-learn | | | | |
| statsmodels | | | | |

---

## Getting Started

### Prerequisites

- Python 3.10+

### Installation
```bash
git clone https://github.com/Santdav/linear-regression-comparison.git
cd linear-regression-comparison
pip install -r requirements.txt
```

### Requirements
```
numpy
pandas
scikit-learn
statsmodels
matplotlib
jupyter
```

### Running

Open any notebook in `notebooks/` to run a specific method, or run all
scripts from `src/` directly:
```bash
python src/gradient_descent.py
python src/normal_equation.py
python src/sklearn_model.py
python src/statsmodels_model.py
```

---

## Key Takeaways

- Gradient descent and the normal equation converge to the same solution,
  but differ in compute cost and transparency.
- scikit-learn is the practical default for production pipelines.
- statsmodels shines when statistical inference on coefficients matters.
- The normal equation becomes numerically unstable for very large feature
  sets; gradient descent scales better in those cases.

---

## Author

Santiago — Systems Engineering Student, Universidad Metropolitana  
[GitHub: Santdav](https://github.com/Santdav)

---

## License

MIT