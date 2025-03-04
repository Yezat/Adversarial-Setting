import numpy as np
import numba as nb
import ctypes

COMPUTE_APPROXIMATE_PROXIMAL = False

DSO = ctypes.CDLL("./numerics/brentq.so")


# Add typing information
brentq_c_func = DSO.brentq
brentq_c_func.restype = ctypes.c_double
brentq_c_func.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]


@nb.jit
def evaluate_proximal(V: float, y: float, epsilon_term: float, w: float) -> float:
    if COMPUTE_APPROXIMATE_PROXIMAL:
        return w + y * V * np.exp(-y * w + epsilon_term) / (
            1 + np.exp(-y * w + epsilon_term)
        )

    if y == 0:
        return w

    w_prime = w - epsilon_term / y
    z = brentq_c_func(-50000000, 50000000, 10e-10, 10e-10, 500, y, V, w_prime)
    return z + epsilon_term / y
