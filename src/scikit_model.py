from lin_reg_strategy import LinearRegressionStrategy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ScikitLinearRegression(LinearRegressionStrategy):
    
    def __init__(self):
        super().__init__()
        self._model = LinearRegression()

    def fit(self, X:np.ndarray, y:np.ndarray):
        self._model.fit(X,y)

    def predict(self, X):
        return self._model.predict(X)
    
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def r_squared(y_true, y_pred):
        return r2_score(y_true,y_pred)