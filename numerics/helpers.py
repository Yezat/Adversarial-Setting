import numpy as np
import numba as nb
from scipy.special import erfc


def sigmoid(x):
    out = np.zeros_like(x)
    idx = x <= 0
    out[idx] = np.exp(x[idx]) / (1 + np.exp(x[idx]))
    idx = x > 0
    out[idx] = 1 / (1 + np.exp(-x[idx]))
    return out


def log1pexp(x):
    out = np.zeros_like(x)
    idx0 = x <= -37
    out[idx0] = np.exp(x[idx0])
    idx1 = (x > -37) & (x <= -2)
    out[idx1] = np.log1p(np.exp(x[idx1]))
    idx2 = (x > -2) & (x <= 18)
    out[idx2] = np.log(1.0 + np.exp(x[idx2]))
    idx3 = (x > 18) & (x <= 33.3)
    out[idx3] = x[idx3] + np.exp(-x[idx3])
    idx4 = x > 33.3
    out[idx4] = x[idx4]
    return out


def adversarial_loss(y, z, epsilon_term):
    return log1pexp(-y * z + epsilon_term)


@nb.vectorize("float64(float64, float64, float64)")
def gaussian(x: float, mean: float = 0, var: float = 1) -> float:
    """
    Gaussian measure
    """
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


@nb.vectorize("float64(float64)")
def stable_cosh_numba(x: float) -> float:
    # technically evaluates 1/(2 * cosh(x))
    if x <= 0:
        return np.exp(x) / (1 + np.exp(2 * x))
    else:
        return np.exp(-x) / (1 + np.exp(-2 * x))


@nb.vectorize("float64(float64, float64, float64)")
def numba_second_derivative_loss(y: float, z: float, epsilon_term: float) -> float:
    return y**2 * stable_cosh_numba(0.5 * y * z - 0.5 * epsilon_term) ** (2)


@nb.vectorize("float64(float64)")
def log1pexp_numba(x: float) -> float:
    if x <= -37:
        return np.exp(x)
    elif -37 < x <= -2:
        return np.log1p(np.exp(x))
    elif -2 < x <= 18:
        return np.log(1.0 + np.exp(x))
    elif 18 < x <= 33.3:
        return x + np.exp(-x)
    else:
        return x


def sigma_star(x):
    """
    Returns 0.5 * erfc(-x/sqrt(2))
    """
    return 0.5 * erfc(-x / np.sqrt(2))
