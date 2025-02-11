import numpy as np
from numerics.helpers import sigmoid


def predict_erm(X, weights):
    return np.sign(predict_erm_probability(X, weights) - 0.5)


def predict_erm_probability(X, weights):
    argument = X @ weights / np.sqrt(X.shape[1])

    return sigmoid(argument)
