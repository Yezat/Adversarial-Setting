import numpy as np
from model.data import DataModel


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


def generate_data(model: DataModel, n: int, tau: float, seed=42) -> DataSet:
    """
    Generates n training data X, y, and test data X_test, y_test and a teacher weight vector w using noise-level tau.

    Args:
        n (int): Number of training samples.
        tau (float): Noise level.
    """
    rng = np.random.default_rng(seed=seed)

    θ = rng.multivariate_normal(np.zeros(model.d), model.Σ_θ, 1, method="cholesky")[0]

    X = rng.multivariate_normal(np.zeros(model.d), model.Σ_x, n, method="cholesky")
    y = np.sign(X @ θ / np.sqrt(model.d) + tau * np.random.normal(0, 1, (n,)))

    X_test = rng.multivariate_normal(
        np.zeros(model.d), model.Σ_x, 10000, method="cholesky"
    )
    y_test = np.sign(
        X_test @ θ / np.sqrt(model.d) + tau * np.random.normal(0, 1, (10000,))
    )

    return DataSet(X, y, X_test, y_test, θ)
