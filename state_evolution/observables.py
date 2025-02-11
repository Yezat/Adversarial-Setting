import numpy as np
from scipy.integrate import quad
from state_evolution.overlaps import Overlaps
from numerics.helpers import gaussian, log1pexp_numba, sigma_star
from state_evolution.proximal import evaluate_proximal
from util.task import Task
from model.data import DataModel
from scipy.special import erfc, erf, logit, owens_t

import numba as nb
import math


@nb.njit
def _training_error_integrand(
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

    # z_out_0 and f_out_0 simplify together as the erfc cancels. See computation
    w = np.sqrt(q) * xi

    proximal = evaluate_proximal(sigma, y, epsilon * np.sqrt(P), w)

    activation = np.sign(proximal)

    return z_0 * gaussian(xi, 0, 1) * (activation != y)


@nb.njit
def _training_loss_integrand(
    xi: float,
    y: float,
    q: float,
    m: float,
    ρ: float,
    tau: float,
    epsilon: float,
    P: float,
    sigma: float,
) -> float:
    w = np.sqrt(q) * xi
    z_0 = math.erfc(
        ((-y * m * xi) / np.sqrt(q)) / np.sqrt(2 * (tau**2 + (ρ - m**2 / q)))
    )

    proximal = evaluate_proximal(sigma, y, epsilon * np.sqrt(P), w)

    loss = log1pexp_numba(-y * proximal + epsilon * P)

    return z_0 * loss * gaussian(xi, 0, 1)


@nb.njit
def _test_loss_integrand(
    xi: float,
    y: float,
    m: float,
    q: float,
    ρ: float,
    tau: float,
    epsilon: float,
    A: float,
) -> float:
    e = m * m / (ρ * q)
    w_0 = np.sqrt(ρ * e) * xi
    V_0 = ρ * (1 - e)

    z_0 = math.erfc((-y * w_0) / np.sqrt(2 * (tau**2 + V_0)))

    w = np.sqrt(q) * xi

    loss_value = log1pexp_numba(-y * w + epsilon * A)

    return z_0 * gaussian(xi, 0, 1) * loss_value


def training_loss_logistic_with_regularization(
    task: Task, overlaps: Overlaps, data_model: DataModel, int_lims: float
):
    return (
        training_loss(task, overlaps, data_model, int_lims)
        + (task.lam / (2 * task.alpha)) * overlaps.q
    )


def training_loss(
    task: Task, overlaps: Overlaps, data_model: DataModel, int_lims: float
):
    I1 = quad(
        lambda xi: _training_loss_integrand(
            xi,
            1,
            overlaps.q,
            overlaps.m,
            data_model.ρ,
            task.tau,
            task.epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    I2 = quad(
        lambda xi: _training_loss_integrand(
            xi,
            -1,
            overlaps.q,
            overlaps.m,
            data_model.ρ,
            task.tau,
            task.epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    return (I1 + I2) / 2


@staticmethod
def training_error(
    task: Task, overlaps: Overlaps, data_model: DataModel, int_lims: float
):
    Iplus = quad(
        lambda xi: _training_error_integrand(
            xi,
            1,
            overlaps.m,
            overlaps.q,
            data_model.ρ,
            task.tau,
            task.epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    Iminus = quad(
        lambda xi: _training_error_integrand(
            xi,
            -1,
            overlaps.m,
            overlaps.q,
            data_model.ρ,
            task.tau,
            task.epsilon,
            overlaps.P,
            overlaps.sigma,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    return (Iplus + Iminus) * 0.5


@staticmethod
def test_loss(
    task: Task,
    overlaps: Overlaps,
    data_model: DataModel,
    epsilon: float,
    int_lims: float,
):
    Iplus = quad(
        lambda xi: _test_loss_integrand(
            xi,
            1,
            overlaps.m,
            overlaps.q,
            data_model.ρ,
            task.tau,
            epsilon,
            overlaps.A,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    Iminus = quad(
        lambda xi: _test_loss_integrand(
            xi,
            -1,
            overlaps.m,
            overlaps.q,
            data_model.ρ,
            task.tau,
            epsilon,
            overlaps.A,
        ),
        -int_lims,
        int_lims,
        limit=500,
    )[0]
    return (Iplus + Iminus) * 0.5


def generalization_error(ρ, m, q, tau):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt((ρ + tau**2) * q)) / np.pi


@nb.njit
def teacher_error(nu: float, tau: float, ρ: float) -> float:
    return np.exp(-(nu**2) / (2 * ρ)) * (1 + math.erf(nu / (np.sqrt(2) * tau)))


def compute_data_model_angle(data_model: DataModel, overlaps: Overlaps, tau):
    L = (
        overlaps.sigma_hat * data_model.spec_Σ_x
        + overlaps.P_hat * data_model.spec_Σ_δ
        + overlaps.N_hat * np.ones(data_model.d)
    )
    return np.sum(data_model.spec_ΦΦT / L) / np.sqrt(
        (data_model.d * tau**2 + data_model.d * data_model.ρ)
        * np.sum(data_model.spec_ΦΦT * data_model.spec_Σ_x / L**2)
    )


def compute_data_model_attackability(data_model: DataModel, overlaps: Overlaps):
    L = (
        overlaps.sigma_hat * data_model.spec_Σ_x
        + overlaps.P_hat * data_model.spec_Σ_δ
        + overlaps.N_hat * np.ones(data_model.d)
    )
    return np.sum(data_model.spec_ΦΦT * data_model.spec_Σ_ν / L**2) / np.sqrt(
        np.sum(data_model.spec_ΦΦT / L**2)
        * np.sum(data_model.spec_ΦΦT * data_model.spec_Σ_x / L**2)
    )


def asymptotic_adversarial_generalization_error(
    data_model: DataModel, overlaps: Overlaps, epsilon, tau
):
    angle = compute_data_model_angle(data_model, overlaps, tau)
    attackability = compute_data_model_attackability(data_model, overlaps)

    a = angle / np.sqrt(1 - angle**2)

    b = epsilon * attackability

    owen = 2 * owens_t(a * b, 1 / a)

    erferfc = 0.5 * erf(b / np.sqrt(2)) * erfc(-a * b / np.sqrt(2))

    gen_error = owen + erferfc

    return gen_error


def adversarial_generalization_error_overlaps_teacher(
    overlaps: Overlaps, task: Task, data_model: DataModel, epsilon: float
):
    # if tau is not zero, we can use the simpler formula
    if task.tau >= 1e-10:
        integral = quad(
            lambda nu: teacher_error(nu, task.tau, data_model.ρ),
            epsilon * overlaps.F / np.sqrt(overlaps.N),
            np.inf,
        )[0]
        return 1 - integral / (np.sqrt(2 * np.pi * data_model.ρ))

    return erf(epsilon * overlaps.F / np.sqrt(2 * data_model.ρ * overlaps.N))


def adversarial_generalization_error_overlaps(
    overlaps: Overlaps, task: Task, data_model: DataModel, epsilon: float
):
    a = overlaps.m / np.sqrt(
        (overlaps.q * (data_model.ρ + task.tau**2) - overlaps.m**2)
    )

    b = epsilon * np.sqrt(overlaps.A) / np.sqrt(overlaps.q)

    owen = 2 * owens_t(a * b, 1 / a)

    erferfc = 0.5 * erf(b / np.sqrt(2)) * erfc(-a * b / np.sqrt(2))

    gen_error = owen + erferfc

    # angle = generalization_error(data_model.ρ, overlaps.m, overlaps.q, task.tau)

    def integrand(xi):
        return (
            (1 / np.sqrt(np.pi * 2))
            * np.exp(-(xi**2) / (2 * data_model.ρ))
            * (1 + erf(xi / (np.sqrt(2) * task.tau)))
            * gaussian(xi, 0, 1)
        )

    return gen_error


def first_term_fair_error(overlaps, data_model, gamma, epsilon):
    V = (data_model.ρ) * overlaps.q - overlaps.m**2
    gamma_max = gamma + epsilon * overlaps.F / np.sqrt(overlaps.N)

    # first term
    def erfc_term(nu):
        return np.exp((-(nu**2)) / (2 * (data_model.ρ))) * erfc(
            (overlaps.F * overlaps.m * nu + overlaps.A * (data_model.ρ) * (gamma - nu))
            / (overlaps.F * np.sqrt(2 * (data_model.ρ) * V))
        )

    def erf_term(nu):
        return np.exp((-(nu**2)) / (2 * (data_model.ρ))) * (
            1
            + erf(
                (
                    overlaps.F * overlaps.m * nu
                    - overlaps.A * (data_model.ρ) * (nu + gamma)
                )
                / (overlaps.F * np.sqrt(2 * (data_model.ρ) * V))
            )
        )

    first_term = quad(lambda nu: erfc_term(nu), gamma, gamma_max, limit=500)[0]
    first_term += quad(lambda nu: erf_term(nu), -gamma_max, -gamma, limit=500)[0]
    first_term /= 2 * np.sqrt(2 * np.pi * (data_model.ρ))
    return first_term


def second_term_fair_error(overlaps, data_model, gamma, epsilon):
    V = (data_model.ρ) * overlaps.q - overlaps.m**2
    gamma_max = gamma + epsilon * overlaps.F / np.sqrt(overlaps.N)

    # second term
    def second_integral(nu):
        return erfc(
            (
                -epsilon * overlaps.A * (data_model.ρ)
                + np.sqrt(overlaps.N) * overlaps.m * nu
            )
            / np.sqrt(overlaps.N * 2 * (data_model.ρ) * V)
        ) * np.exp(-((nu) ** 2) / (2 * (data_model.ρ)))

    result2 = quad(lambda nu: second_integral(nu), gamma_max, np.inf, limit=500)
    second_term = result2[0]
    second_term /= np.sqrt(2 * np.pi * (data_model.ρ))

    return second_term


def third_term_fair_error(overlaps, data_model, gamma, epsilon):
    V = (data_model.ρ) * overlaps.q - overlaps.m**2
    # gamma_max = gamma + epsilon * overlaps.F / np.sqrt(overlaps.N)

    # third term
    def third_integral(nu):
        return np.exp(-((nu) ** 2) / (2 * (data_model.ρ))) * erfc(
            overlaps.m * nu / np.sqrt(2 * (data_model.ρ) * V)
        )

    result3 = quad(lambda nu: third_integral(nu), 0, gamma, limit=500)
    third_term = result3[0]
    third_term /= np.sqrt(2 * np.pi * (data_model.ρ))

    return third_term


def fair_adversarial_error_overlaps(overlaps, data_model, gamma, epsilon, logger=None):
    first_term = first_term_fair_error(overlaps, data_model, gamma, epsilon)

    second_term = second_term_fair_error(overlaps, data_model, gamma, epsilon)

    third_term = third_term_fair_error(overlaps, data_model, gamma, epsilon)

    return first_term + second_term + third_term


def overlap_calibration(ρ, p, m, q_erm, tau, debug=False):
    """
    Analytical calibration for a given probability p, overlaps m and q and a given noise level tau
    Given by equation 23 in 2202.03295.
    Returns the calibration value.

    p: probability between 0 and 1
    m: overlap between teacher and student
    q_erm: student overlap
    tau: noise level
    """
    logi = logit(p)
    m_q_ratio = m / (q_erm)

    num = (logi) * m_q_ratio
    if debug:
        print("tau", tau, "m**2", m**2, "q_erm", q_erm, "m**2/q_erm", m**2 / q_erm)
    denom = np.sqrt(ρ - (m**2) / (q_erm) + (tau) ** 2)
    if debug:
        print("logi", logi, "m_q_ratio", m_q_ratio, "num", num, "denom", denom)
    return p - sigma_star((num) / (denom))
