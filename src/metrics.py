import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the R^2 score."""
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    return float(1 - (ss_res / ss_tot))
