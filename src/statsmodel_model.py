from src.lin_reg_strategy import LinearRegressionStrategy
import numpy as np
from statsmodels.api import OLS, add_constant

class StatsModelsLinearRegression(LinearRegressionStrategy):
    def __init__(self):
        self.model = None
        self.results = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_with_constant = add_constant(X)
        self.model = OLS(y, X_with_constant)
        self.results = self.model.fit()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Model must be fitted before predicting.")
        X_with_constant = add_constant(X, has_constant='add')
        return self.results.predict(X_with_constant)

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def name(self):
        return "Stats model"