import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import add_dummy_feature
from lin_reg_strategy import LinearRegressionStrategy

class SklearnLinearRegression(LinearRegressionStrategy):
    def __init__(self):
        # We set fit_intercept=False because we are adding the constant manually
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Adds a column of ones to the beginning of the array
        X_with_constant = add_dummy_feature(X, value=1.0)
        self.model.fit(X_with_constant, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Ensure the input features also have the constant column
        X_with_constant = add_dummy_feature(X, value=1.0)
        return self.model.predict(X_with_constant)

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)

    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)