"""
This module contains losses and gradients for the coefficient logistic regression problem.
Loss-gradient receives additional coefficients through V to approximate the logistic problem.
"""

import numpy as np
from numerics.helpers import log1pexp, sigmoid
import logging


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

    optimal_attack = epsilon / np.sqrt(n_features) * np.sqrt(wSw)

    margins = y * raw_prediction

    shifted_margins = margins - optimal_attack

    # mask the shifted margins where they are positive
    mask_positive = shifted_margins > 0

    # compute corresponding subsets
    lambda_twiddle_1 = V["lambda_twiddle_1"]
    lambda_twiddle_2 = V["lambda_twiddle_2"]

    logging.info(V)

    loss = compute_coefficient_logistic_loss(margins)
    loss += np.log(2) + lambda_twiddle_1 * np.sqrt(wSw) + lambda_twiddle_2 * wSw
    loss += l2_reg_strength * (weights @ covariance_prior @ weights)

    gradient_per_sample = compute_coefficient_logistic_gradient(margins)

    derivative_optimal_attack = (
        epsilon / np.sqrt(n_features) * Σ_δ @ weights / (np.sqrt(wSw))
    )

    adv_grad_summand = np.outer(gradient_per_sample, -derivative_optimal_attack).sum(
        axis=0
    )
    adv_grad_summand = 0
    negative_adv_grad_summand = (
        lambda_twiddle_1 * 0.5 / np.sqrt(wSw) * (Σ_δ + Σ_δ.T) @ weights
    ) + lambda_twiddle_2 * (Σ_δ + Σ_δ.T) @ weights

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
