import numpy as np
from scipy.integrate import quad
from state_evolution.overlaps import Overlaps
from numerics.helpers import gaussian, log1pexp_numba
from numerics.proximal import evaluate_proximal
from util.task import Task
from scipy.special import erfc, erf, owens_t
from model.data import DataModel

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

    return gen_error
