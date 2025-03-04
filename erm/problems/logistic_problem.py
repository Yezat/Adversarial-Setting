import numpy as np
from numerics.helpers import log1pexp, sigmoid


def logistic_loss_gradient(
    coef,
    X,
    y,
    l2_reg_strength,
    epsilon,
    covariance_prior,
    Σ_δ,
    V=None,
):
    n_features = X.shape[1]
    weights = coef
    raw_prediction = X @ weights / np.sqrt(n_features)

    l2_reg_strength /= 2

    wSw = weights.dot(Σ_δ @ weights)

    optimal_attack = epsilon / np.sqrt(n_features) * np.sqrt(wSw)

    loss = compute_logistic_loss(raw_prediction, optimal_attack, y)
    loss = loss.sum()
    loss += l2_reg_strength * (weights @ covariance_prior @ weights)

    epsilon_gradient_per_sample, gradient_per_sample = compute_logistic_gradient(
        raw_prediction, optimal_attack, y
    )

    derivative_optimal_attack = (
        epsilon / np.sqrt(n_features) * Σ_δ @ weights / (np.sqrt(wSw))
    )

    adv_grad_summand = np.outer(
        epsilon_gradient_per_sample, derivative_optimal_attack
    ).sum(axis=0)

    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert (
            np.linalg.norm(adv_grad_summand) == 0
        ), f"derivative_optimal_attack {np.linalg.norm(derivative_optimal_attack)}, epsilon_gradient_per_sample {np.linalg.norm(epsilon_gradient_per_sample)}"

    grad = np.empty_like(coef, dtype=weights.dtype)
    grad[:n_features] = (
        X.T @ gradient_per_sample / np.sqrt(n_features)
        + l2_reg_strength * (covariance_prior + covariance_prior.T) @ weights
        + adv_grad_summand
    )

    return loss, grad


def compute_logistic_loss(z, e, y):
    return -y * z + y * e + (1 - y) * log1pexp(z + e) + y * log1pexp(z - e)


def compute_logistic_gradient(z, e, y):
    opt_attack_term = (1 - y) * sigmoid(z + e) + y * sigmoid(-z + e)
    data_term = (1 - y) * sigmoid(z + e) - y * sigmoid(-z + e)
    return opt_attack_term, data_term
