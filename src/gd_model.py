import numpy as np
from typing import Optional
from src.metrics import mean_squared_error, r2_score
from lin_reg_strategy import LinearRegressionStrategy

class GradientDescentLinearRegression(LinearRegressionStrategy):
    """Linear Regression implementation using Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta: Optional[np.ndarray] = None

        self.X = None
        self.Y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the data using Gradient Descent."""

        self.X = X
        self.Y = y

        m = X.shape[0]
        # Add bias term
        X_b = np.c_[np.ones((m, 1)), X]

        # Initialize theta randomly
        self.theta = np.random.randn(X_b.shape[1], 1)

        # Reshape y to (m, 1)
        y_reshaped = y.reshape(-1, 1)

        # Gradient Descent loop
        for _ in range(self.n_iterations):
            gradients = 2 / m * X_b.T.dot(X_b.dot(self.theta) - y_reshaped)
            self.theta -= self.learning_rate * gradients

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for the given data."""
        if self.theta is None:
            raise ValueError("Model not fitted yet.")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta).flatten()

    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))

    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the R^2 score."""
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        return float(1 - (ss_res / ss_tot))