import numpy as np
from typing import Optional


class EquationLinearRegression:
    """Linear Regression implementation using the Normal Equation."""

    def __init__(self) -> None:
        self.theta: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the data using the Normal Equation."""
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

