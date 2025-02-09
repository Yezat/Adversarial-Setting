import numpy as np


class DataSet:
    """
    A class to store the data and the teacher weight vector.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        θ: np.ndarray,
    ) -> None:
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.X_test: np.ndarray = X_test
        self.y_test: np.ndarray = y_test
        self.θ: np.ndarray = θ
