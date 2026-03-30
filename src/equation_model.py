import numpy as np
from typing import Optional
from lin_reg_strategy import LinearRegressionStrategy

class EquationLinearRegression(LinearRegressionStrategy):
    """Linear Regression implementation using the Normal Equation."""

    def __init__(self) -> None:
        self.theta: Optional[np.ndarray] = None
        self.X = None
        self.Y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the data using the Normal Equation."""

        self.X = X
        self.Y = y
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal Equation: (X^T * X)^-1 * X^T * y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for the given data."""
        if self.theta is None:
            raise ValueError("Model not fitted yet.")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
    
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))

    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the R^2 score."""
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        return float(1 - (ss_res / ss_tot))

