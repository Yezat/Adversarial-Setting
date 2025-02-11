import ctypes
import numpy as np
import numba as nb

# if ./brentq.so exists load it, otherwise load ../brentq.so
try:
    DSO = ctypes.CDLL("./brentq.so")
except OSError:
    raise OSError(
        "Could not load the shared object file. Please build this project before running the code."
    )

# Add typing information
c_func = DSO.brentq
c_func.restype = ctypes.c_double
c_func.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

COMPUTE_APPROXIMATE_PROXIMAL = False


@nb.njit
def evaluate_proximal(V: float, y: float, epsilon_term: float, w: float) -> float:
    if COMPUTE_APPROXIMATE_PROXIMAL:
        return w + y * V * np.exp(-y * w + epsilon_term) / (
            1 + np.exp(-y * w + epsilon_term)
        )

    if y == 0:
        return w

    w_prime = w - epsilon_term / y
    z = c_func(-50000000, 50000000, 10e-10, 10e-10, 500, y, V, w_prime)
    return z + epsilon_term / y
