"""
This module contains losses and gradients for the coefficient logistic regression problem.
Loss-gradient receives additional coefficients through V to approximate the logistic problem.
"""

import numpy as np
from numerics.helpers import log1pexp, sigmoid


def coefficient_logistic_loss_gradient(
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
    nww = np.sqrt(weights @ weights)

    optimal_attack = epsilon / np.sqrt(n_features) * wSw / nww

    margins = y * raw_prediction

    shifted_margins = margins - optimal_attack

    # mask the shifted margins where they are positive
    mask_positive = shifted_margins > 0

    # compute corresponding subsets
    empirical_e_lam_1 = V["empirical_e_lam_1"]
    direct_cross_term = V["direct_cross_term"]

    loss = compute_coefficient_logistic_loss(margins)
    loss += np.log(2) + empirical_e_lam_1 * wSw / nww + direct_cross_term * wSw
    loss += l2_reg_strength * (weights @ covariance_prior @ weights)

    gradient_per_sample = compute_coefficient_logistic_gradient(margins)

    derivative_optimal_attack = (
        epsilon
        / np.sqrt(n_features)
        * (2 * Σ_δ @ weights / nww - (wSw / nww**3) * weights)
    )

    adv_grad_summand = np.outer(gradient_per_sample, -derivative_optimal_attack).sum(
        axis=0
    )
    adv_grad_summand = 0
    negative_adv_grad_summand = (
        empirical_e_lam_1 / nww * weights
    ) + direct_cross_term * (Σ_δ + Σ_δ.T) @ weights

    positive_data = X[mask_positive]
    positive_data = X

    positive_labels = y[mask_positive]
    positive_labels = y

    label_data_product = (
        positive_labels[:, np.newaxis] * positive_data / np.sqrt(n_features)
    )
    grad_contribution = label_data_product.T @ gradient_per_sample

    grad = np.empty_like(coef, dtype=weights.dtype)
    grad[:n_features] = (
        grad_contribution
        + l2_reg_strength * (covariance_prior + covariance_prior.T) @ weights
        + adv_grad_summand
        + negative_adv_grad_summand
    )

    return loss, grad


def compute_coefficient_logistic_loss(shifted_margins_positive):
    first_part = np.sum(log1pexp(-shifted_margins_positive))

    return first_part


def compute_coefficient_logistic_gradient(shifted_margins_positive):
    positive_part = -sigmoid(-shifted_margins_positive)
    return positive_part
