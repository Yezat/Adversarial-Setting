import numpy as np
from numerics.helpers import adversarial_loss


def error(y, yhat):
    return 0.25 * np.mean((y - yhat) ** 2)


def adversarial_error_fgm(y, Xtest, w_gd, epsilon, Σ_ν):
    d = Xtest.shape[1]
    wSw = w_gd.dot(Σ_ν @ w_gd)
    nww = np.sqrt(w_gd @ w_gd)

    return error(
        y, np.sign(Xtest @ w_gd / np.sqrt(d) - y * epsilon / np.sqrt(d) * wSw / nww)
    )


def adversarial_error(y, Xtest, w_gd, epsilon, Σ_ν):
    d = Xtest.shape[1]
    wSw = w_gd.dot(Σ_ν @ w_gd)

    return error(
        y, np.sign(Xtest @ w_gd / np.sqrt(d) - y * epsilon / np.sqrt(d) * np.sqrt(wSw))
    )


def compute_boundary_loss(y, Xtest, epsilon, Σ_δ, w_gd, l2_reg_strength):
    d = Xtest.shape[1]
    wSw = w_gd.dot(Σ_δ @ w_gd)

    optimal_attack = epsilon / np.sqrt(d) * np.sqrt(wSw)

    raw_prediction = Xtest @ w_gd / np.sqrt(d)

    # compute y * raw_prediction elementwise and sum over all samples
    y_raw_prediction = y * raw_prediction
    y_raw_prediction_sum = y_raw_prediction.sum()

    boundary_loss = y_raw_prediction_sum * optimal_attack * l2_reg_strength

    # assert boundary_loss to be a scalar
    assert np.isscalar(boundary_loss)

    return boundary_loss


def compute_boundary_loss_fgm(y, Xtest, epsilon, Σ_δ, w_gd, l2_reg_strength):
    d = Xtest.shape[1]
    wSw = w_gd.dot(Σ_δ @ w_gd)
    nww = np.sqrt(w_gd @ w_gd)

    optimal_attack = epsilon / np.sqrt(d) * wSw / nww

    raw_prediction = Xtest @ w_gd / np.sqrt(d)

    # compute y * raw_prediction elementwise and sum over all samples
    y_raw_prediction = y * raw_prediction
    y_raw_prediction_sum = y_raw_prediction.sum()

    boundary_loss = y_raw_prediction_sum * optimal_attack * l2_reg_strength

    # assert boundary_loss to be a scalar
    assert np.isscalar(boundary_loss)

    return boundary_loss


def fair_adversarial_error_erm(
    X_test, w_gd, teacher_weights, epsilon, gamma, data_model
):
    d = X_test.shape[1]

    N = w_gd @ w_gd
    A = w_gd.dot(data_model.Σ_ν @ w_gd)
    F = w_gd.dot(data_model.Σ_ν @ teacher_weights)
    teacher_activation = X_test @ teacher_weights / np.sqrt(d)
    student_activation = X_test @ w_gd / np.sqrt(d)

    y = np.sign(teacher_activation)

    gamma_constraint_argument = y * teacher_activation - epsilon * F / np.sqrt(N * d)

    # first term
    y_first = np.zeros_like(y)
    y_gamma = np.zeros_like(y)
    moved_argument = student_activation + A / F * (y * gamma - teacher_activation)
    y_gamma_t = np.sign(moved_argument)
    mask_gamma_smaller = (
        y * teacher_activation < gamma + epsilon * F / np.sqrt(N * d)
    ) & (y * teacher_activation > gamma)
    y_gamma[mask_gamma_smaller] = y_gamma_t[mask_gamma_smaller]
    y_first[mask_gamma_smaller] = y[mask_gamma_smaller]
    first_error = error(y_first, y_gamma)

    # second term
    y_second = np.zeros_like(y)
    y_max = np.zeros_like(y)
    mask_gamma_bigger = (gamma_constraint_argument >= gamma) & (
        y * teacher_activation > gamma
    )
    y_max_t = np.sign(student_activation - y * epsilon * A / np.sqrt(N * d))
    y_max[mask_gamma_bigger] = y_max_t[mask_gamma_bigger]
    y_second[mask_gamma_bigger] = y[mask_gamma_bigger]
    second_error = error(y_second, y_max)

    # third term
    y_hat = np.zeros_like(y)
    y_third = np.zeros_like(y)
    mask_last_smaller = (y * teacher_activation <= gamma) & (y * teacher_activation > 0)
    y_hat_t = np.sign(X_test @ w_gd)
    y_hat[mask_last_smaller] = y_hat_t[mask_last_smaller]
    y_third[mask_last_smaller] = y[mask_last_smaller]
    third_error = error(y_third, y_hat)

    return first_error + second_error + third_error


def logistic_training_loss_with_regularization(
    w, X, y, lam, epsilon, covariance_prior=None
):
    z = X @ w
    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])
    return (
        adversarial_loss(y, z, epsilon / np.sqrt(X.shape[1]), w @ w).sum()
        + 0.5 * lam * w @ covariance_prior @ w
    ) / X.shape[0]


def logistic_training_loss(w, X, y, epsilon, Σ_δ):
    z = X @ w / np.sqrt(X.shape[1])
    attack = epsilon / np.sqrt(X.shape[1]) * (w.dot(Σ_δ @ w) / np.sqrt(w @ w))
    return (adversarial_loss(y, z, attack).sum()) / X.shape[0]
