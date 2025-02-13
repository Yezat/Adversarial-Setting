import numpy as np
from scipy.integrate import quad
from state_evolution.overlaps import Overlaps
from state_evolution.constants import INT_LIMS
from numerics.helpers import numba_second_derivative_loss, gaussian
from numerics.proximal import evaluate_proximal

import numba as nb
import math


"""
------------------------------------------------------------------------------------------------------------------------
    Hat Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""


@nb.njit
def _m_hat_integrand(
    xi: float,
    y: float,
    m: float,
    q: float,
    ρ: float,
    tau: float,
    epsilon: float,
    P: float,
    sigma: float,
) -> float:
    e = m * m / (ρ * q)
    w_0 = np.sqrt(ρ * e) * xi
    V_0 = ρ * (1 - e)

    # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
    w = np.sqrt(q) * xi

    partial_prox = evaluate_proximal(sigma, y, epsilon * np.sqrt(P), w) - w

    return partial_prox * gaussian(w_0, 0, V_0 + tau**2) * gaussian(xi, 0, 1)


def logistic_m_hat_func(
    overlaps: Overlaps,
    ρ: float,
    alpha: float,
    epsilon: float,
    tau: float,
    int_lims: float = 20.0,
):
    Iplus = quad(
        lambda xi: _m_hat_integrand(
            xi,
            1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    Iminus = quad(
        lambda xi: _m_hat_integrand(
            xi,
            -1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    return alpha / overlaps.sigma * (Iplus - Iminus)


@nb.njit
def _q_hat_integrand(
    xi: float,
    y: float,
    m: float,
    q: float,
    ρ: float,
    tau: float,
    epsilon: float,
    P: float,
    sigma: float,
) -> float:
    e = m * m / (ρ * q)
    w_0 = np.sqrt(ρ * e) * xi
    V_0 = ρ * (1 - e)

    z_0 = math.erfc((-y * w_0) / np.sqrt(2 * (tau**2 + V_0)))

    w = np.sqrt(q) * xi

    proximal = evaluate_proximal(sigma, y, epsilon * np.sqrt(P), w)
    partial_proximal = (proximal - w) ** 2

    return z_0 * (partial_proximal / (sigma**2)) * gaussian(xi, 0, 1)


def logistic_q_hat_func(
    overlaps: Overlaps,
    ρ: float,
    alpha: float,
    epsilon: float,
    tau: float,
    int_lims: float = 20.0,
):
    Iplus = quad(
        lambda xi: _q_hat_integrand(
            xi,
            1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    Iminus = quad(
        lambda xi: _q_hat_integrand(
            xi,
            -1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]

    return 0.5 * alpha * (Iplus + Iminus)


"""
Derivative of f_out
"""


@nb.njit
def alternative_derivative_f_out(
    xi: float,
    y: float,
    m: float,
    q: float,
    sigma: float,
    epsilon: float,
    P: float,
) -> float:
    w = np.sqrt(q) * xi

    proximal = evaluate_proximal(sigma, y, epsilon * np.sqrt(P), w)

    second_derivative = numba_second_derivative_loss(y, proximal, epsilon * np.sqrt(P))

    return second_derivative / (
        1 + sigma * second_derivative
    )  # can be seen from aubin (45)


@nb.njit
def _sigma_hat_integrand(
    xi: float,
    y: float,
    m: float,
    q: float,
    ρ: float,
    tau: float,
    epsilon: float,
    P: float,
    sigma: float,
) -> float:
    z_0 = math.erfc(
        ((-y * m) / np.sqrt(q) * xi) / np.sqrt(2 * (tau**2 + (ρ - m**2 / q)))
    )

    derivative_f_out = alternative_derivative_f_out(xi, y, m, q, sigma, epsilon, P)

    return z_0 * (derivative_f_out) * gaussian(xi, 0, 1)


def logistic_sigma_hat_func(
    overlaps: Overlaps,
    ρ: float,
    alpha: float,
    epsilon: float,
    tau: float,
    int_lims: float = 20.0,
):
    Iplus = quad(
        lambda xi: _sigma_hat_integrand(
            xi,
            1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    Iminus = quad(
        lambda xi: _sigma_hat_integrand(
            xi,
            -1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]

    return 0.5 * alpha * (Iplus + Iminus)


@nb.njit
def _P_hat_integrand(
    xi: float,
    y: float,
    m: float,
    q: float,
    ρ: float,
    tau: float,
    epsilon: float,
    P: float,
    sigma: float,
) -> float:
    e = m * m / (ρ * q)
    w_0 = np.sqrt(ρ * e) * xi
    V_0 = ρ * (1 - e)

    z_0 = math.erfc((-y * w_0) / np.sqrt(2 * (tau**2 + V_0)))

    w = np.sqrt(q) * xi

    z_star = evaluate_proximal(sigma, y, epsilon * np.sqrt(P), w)

    m_derivative = -(z_star - w) / sigma

    m_derivative *= -y * epsilon * 0.5 / np.sqrt(P)

    return z_0 * m_derivative * gaussian(xi, 0, 1)


def logistic_P_hat_func(
    overlaps: Overlaps,
    ρ: float,
    alpha: float,
    epsilon: float,
    tau: float,
    int_lims: float = 20.0,
):
    Iplus = quad(
        lambda xi: _P_hat_integrand(
            xi,
            1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    Iminus = quad(
        lambda xi: _P_hat_integrand(
            xi,
            -1,
            overlaps.m,
            overlaps.q,
            ρ,
            tau,
            epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]

    return alpha * (Iplus + Iminus)


"""
------------------------------------------------------------------------------------------------------------------------
    Overlap Equations
------------------------------------------------------------------------------------------------------------------------
"""


def var_hat_func(task, overlaps, data_model):
    overlaps.m_hat = logistic_m_hat_func(
        overlaps,
        data_model.ρ,
        task.alpha,
        task.epsilon,
        task.tau,
        INT_LIMS,
    ) / np.sqrt(data_model.gamma)
    overlaps.q_hat = logistic_q_hat_func(
        overlaps,
        data_model.ρ,
        task.alpha,
        task.epsilon,
        task.tau,
        INT_LIMS,
    )
    overlaps.sigma_hat = logistic_sigma_hat_func(
        overlaps,
        data_model.ρ,
        task.alpha,
        task.epsilon,
        task.tau,
        INT_LIMS,
    )
    overlaps.P_hat = logistic_P_hat_func(
        overlaps,
        data_model.ρ,
        task.alpha,
        task.epsilon,
        task.tau,
        INT_LIMS,
    )

    return overlaps


def var_func(task, overlaps, data_model, slice_from=None, slice_to=None):
    if slice_to is None:
        slice_to = data_model.d
    if slice_from is None:
        slice_from = 0

    Lambda = (
        task.lam * data_model.spec_Σ_ω[slice_from:slice_to]
        + overlaps.sigma_hat * data_model.spec_Σ_x[slice_from:slice_to]
        + overlaps.P_hat * data_model.spec_Σ_δ[slice_from:slice_to]
    )
    H = (
        data_model.spec_Σ_x[slice_from:slice_to] * overlaps.q_hat
        + overlaps.m_hat**2 * data_model.spec_ΦΦT[slice_from:slice_to]
    )

    sigma = np.mean(data_model.spec_Σ_x[slice_from:slice_to] / Lambda)
    q = np.mean((H * data_model.spec_Σ_x[slice_from:slice_to]) / Lambda**2)
    m = (
        overlaps.m_hat
        / np.sqrt(data_model.gamma)
        * np.mean(data_model.spec_ΦΦT[slice_from:slice_to] / Lambda)
    )

    P = np.mean(H * data_model.spec_Σ_δ[slice_from:slice_to] / Lambda**2)

    A = np.mean(H * data_model.spec_Σ_ν[slice_from:slice_to] / Lambda**2)
    F = (
        0.5
        * overlaps.m_hat
        * np.mean(data_model.spec_FTerm[slice_from:slice_to] / Lambda)
    )

    return m, q, sigma, A, P, F
