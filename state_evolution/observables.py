import numpy as np
from scipy.integrate import quad
from state_evolution.constants import SEProblemType
from state_evolution.overlaps import Overlaps
from numerics.helpers import sigma_star
from util.task import Task
from model.data import DataModel
from scipy.special import erfc, erf, logit, owens_t
from state_evolution.logistic.observables import (
    training_error,
    training_loss,
    test_loss,
    adversarial_generalization_error_overlaps,
)
from state_evolution.logistic_fgm.observables_fgm import (
    training_error as training_error_fgm,
    training_loss as training_loss_fgm,
    test_loss as test_loss_fgm,
    adversarial_generalization_error_overlaps as adversarial_generalization_error_overlaps_fgm,
)

import numba as nb
import math


MAP_PROBLEM_TYPE_TRAINING_ERROR = {
    SEProblemType.Logistic: training_error,
    SEProblemType.LogisticFGM: training_error_fgm,
}

MAP_PROBLEM_TYPE_TRAINING_LOSS = {
    SEProblemType.Logistic: training_loss,
    SEProblemType.LogisticFGM: training_loss_fgm,
}

MAP_PROBLEM_TYPE_TEST_LOSS = {
    SEProblemType.Logistic: test_loss,
    SEProblemType.LogisticFGM: test_loss_fgm,
}

MAP_PROBLEM_TYPE_ADV_GEN_ERR_OVERLAP = {
    SEProblemType.Logistic: adversarial_generalization_error_overlaps,
    SEProblemType.LogisticFGM: adversarial_generalization_error_overlaps_fgm,
}


def training_loss_logistic_with_regularization(
    task: Task, overlaps: Overlaps, data_model: DataModel, int_lims: float
):
    return (
        MAP_PROBLEM_TYPE_TRAINING_LOSS[task.se_problem_type](
            task, overlaps, data_model, int_lims
        )
        + (task.lam / (2 * task.alpha)) * overlaps.q
    )


def generalization_error(ρ, m, q, tau):
    """
    Returns the generalization error in terms of the overlaps
    """
    return np.arccos(m / np.sqrt((ρ + tau**2) * q)) / np.pi


@nb.njit
def teacher_error(nu: float, tau: float, ρ: float) -> float:
    return np.exp(-(nu**2) / (2 * ρ)) * (1 + math.erf(nu / (np.sqrt(2) * tau)))


def compute_data_model_angle(data_model: DataModel, overlaps: Overlaps, tau):
    L = overlaps.sigma_hat * data_model.spec_Σ_x + overlaps.P_hat * data_model.spec_Σ_δ
    return np.sum(data_model.spec_ΦΦT / L) / np.sqrt(
        (data_model.d * tau**2 + data_model.d * data_model.ρ)
        * np.sum(data_model.spec_ΦΦT * data_model.spec_Σ_x / L**2)
    )


def compute_data_model_attackability(data_model: DataModel, overlaps: Overlaps):
    L = overlaps.sigma_hat * data_model.spec_Σ_x + overlaps.P_hat * data_model.spec_Σ_δ
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
    raise NotImplementedError
    # if tau is not zero, we can use the simpler formula
    if task.tau >= 1e-10:
        integral = quad(
            lambda nu: teacher_error(nu, task.tau, data_model.ρ),
            epsilon * overlaps.F / np.sqrt(overlaps.N),
            np.inf,
        )[0]
        return 1 - integral / (np.sqrt(2 * np.pi * data_model.ρ))

    return erf(epsilon * overlaps.F / np.sqrt(2 * data_model.ρ * overlaps.N))


def first_term_fair_error(overlaps, data_model, gamma, epsilon) -> float:
    V = (data_model.ρ) * overlaps.q - overlaps.m**2
    gamma_max = gamma + epsilon * overlaps.F / np.sqrt(overlaps.N)

    # first term
    def erfc_term(nu) -> float:
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


def fair_adversarial_error_overlaps(overlaps, data_model, gamma, epsilon):
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
