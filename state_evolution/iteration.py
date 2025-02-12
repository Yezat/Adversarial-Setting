import logging
from util.task import Task
from model.data import DataModel
from state_evolution.overlaps import Overlaps
from state_evolution.constants import TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE
from state_evolution.logistic_integrals import var_func, var_hat_func


def fixed_point_finder(
    data_model: DataModel,
    task: Task,
    log: bool = True,
):
    overlaps = Overlaps()

    err = 1.0
    iter_nb = 0

    while err > TOL_FPE or iter_nb < MIN_ITER_FPE:
        if iter_nb % 500 == 0 and log:
            logging.info(f"iter_nb: {iter_nb}, err: {err}")
            logging.info(f"error: {err}")

        overlaps = var_hat_func(task, overlaps, data_model)

        new_m, new_q, new_sigma, new_A, new_P, new_F = var_func(
            task, overlaps, data_model
        )

        err = overlaps.update_overlaps(new_m, new_q, new_sigma, new_A, new_P, new_F)

        iter_nb += 1
        if iter_nb > MAX_ITER_FPE:
            logging.info("!!! ------------- MAX ITERATIONS REACHED -------------- !!!")
            raise Exception("fixed_point_finder - reached max_iterations")
    return overlaps
