"""
This module contains losses and gradients for the perturbed logistic regression problem.
Loss-gradient updates the value dict V with the empirically measured coefficients arising from the perturbation.
"""

import numpy as np
from numerics.helpers import log1pexp, sigmoid
import logging


def perturbed_logistic_loss_gradient(
    coef,
    X,
    y,
    l2_reg_strength,
    epsilon,
    covariance_prior,
    Σ_δ,
    V,
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
    shifted_margins_positive = shifted_margins[mask_positive]
    shifted_margins_negative = shifted_margins[~mask_positive]

    n_2 = np.sum(~mask_positive)

    lambda_twiddle_1 = n_2 * epsilon / (2 * np.sqrt(n_features)) - epsilon * np.sum(
        margins[~mask_positive]
    ) / (4 * np.sqrt(n_features))
    lambda_twiddle_2 = n_2 * epsilon**2 / (8 * n_features)

    V["lambda_twiddle_1"] = lambda_twiddle_1
    V["lambda_twiddle_2"] = lambda_twiddle_2

    logging.info(V)

    loss = compute_perturbed_logistic_loss(
        shifted_margins_positive, shifted_margins_negative
    )
    loss += l2_reg_strength * (weights @ covariance_prior @ weights)

    positive_gradient_per_sample, negative_gradient_per_sample = (
        compute_perturbed_logistic_gradient(
            shifted_margins_positive, shifted_margins_negative
        )
    )

    derivative_optimal_attack = (
        epsilon / np.sqrt(n_features) * Σ_δ @ weights / (np.sqrt(wSw))
    )

    positive_adv_grad_summand = np.outer(
        positive_gradient_per_sample, -derivative_optimal_attack
    ).sum(axis=0)
    negative_adv_grad_summand = np.outer(
        negative_gradient_per_sample, -derivative_optimal_attack
    ).sum(axis=0)

    # if epsilon is zero, assert that the norm of adv_grad_summand is zero
    if epsilon == 0:
        assert (
            np.linalg.norm(positive_adv_grad_summand) == 0
        ), f"derivative_optimal_attack {np.linalg.norm(derivative_optimal_attack)}, gradient_per_sample {np.linalg.norm(positive_gradient_per_sample)}"
        assert (
            np.linalg.norm(negative_adv_grad_summand) == 0
        ), f"derivative_optimal_attack {np.linalg.norm(derivative_optimal_attack)}, gradient_per_sample {np.linalg.norm(negative_gradient_per_sample)}"

    positive_data = X[mask_positive]
    negative_data = X[~mask_positive]

    positive_labels = y[mask_positive]
    negative_labels = y[~mask_positive]

    positive_label_data_product = (
        positive_labels[:, np.newaxis] * positive_data / np.sqrt(n_features)
    )
    positive_contribution = positive_label_data_product.T @ positive_gradient_per_sample

    negative_label_data_product = (
        negative_labels[:, np.newaxis] * negative_data / np.sqrt(n_features)
    )
    negative_contribution = negative_label_data_product.T @ negative_gradient_per_sample

    grad = np.empty_like(coef, dtype=weights.dtype)
    grad[:n_features] = (
        positive_contribution
        + negative_contribution
        + l2_reg_strength * (covariance_prior + covariance_prior.T) @ weights
        + positive_adv_grad_summand
        + negative_adv_grad_summand
    )

    return loss, grad


def compute_perturbed_logistic_loss(shifted_margins_positive, shifted_margins_negative):
    first_part = np.sum(log1pexp(-shifted_margins_positive))

    second_part = np.sum(
        np.log(2)
        - 0.5 * shifted_margins_negative
        + (1 / 8) * shifted_margins_negative**2
    )  #  - (1/192)*shifted_margins_negative**4 + (1/2880)*shifted_margins_negative**6)
    # return np.sum(log1pexp(-shifted_margins_positive)) + np.sum( log1pexp(-shifted_margins_negative) )

    return first_part + second_part


def compute_perturbed_logistic_gradient(
    shifted_margins_positive, shifted_margins_negative
):
    positive_part = -sigmoid(-shifted_margins_positive)
    negative_part = (
        -0.5 + shifted_margins_negative / 4
    )  # - (1/48)*shifted_margins_negative**3 + (1/480)*shifted_margins_negative**5
    # negative_part = -sigmoid(-shifted_margins_negative)

    return positive_part, negative_part
