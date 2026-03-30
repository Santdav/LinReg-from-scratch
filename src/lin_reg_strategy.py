from abc import ABC, abstractmethod
import numpy as np

class LinearRegressionStrategy(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) ->None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass