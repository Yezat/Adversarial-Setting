import numpy as np
from erm.prediction import predict_erm, predict_erm_probability


def test_predict_erm():
    X = np.array([[1, 2], [3, 4], [5, 6], [-1, -2]])
    weights = np.array([0.5, 0.5])
    result = predict_erm(X, weights)
    expected = np.array([1, 1, 1, -1])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_predict_erm_probability():
    X = np.array([[1, 2], [3, 4], [5, 6], [-1, -2]])
    weights = np.array([0.5, 0.5])
    result = predict_erm_probability(X, weights)
    expected = np.array([0.74281668, 0.9223615, 0.97994636, 0.25718332])
    assert np.allclose(
        result, expected, atol=1e-5
    ), f"Expected {expected}, but got {result}"
