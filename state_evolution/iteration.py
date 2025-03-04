import logging
from util.task import Task
from model.data import DataModel
from state_evolution.overlaps import Overlaps
from state_evolution.constants import TOL_FPE, MIN_ITER_FPE, MAX_ITER_FPE, SEProblemType
from state_evolution.logistic.logistic_integrals import (
    var_func as logistic_var_func,
    var_hat_func as logistic_var_hat_func,
)

from state_evolution.logistic_fgm.logistic_fgm_integrals import (
    var_func as logistic_fgm_var_func,
    var_hat_func as logistic_fgm_var_hat_func,
)


MAP_PROBLEM_TYPE_VAR_FUNC = {
    SEProblemType.Logistic: logistic_var_func,
    SEProblemType.LogisticFGM: logistic_fgm_var_func,
}

MAP_PROBLEM_TYPE_VAR_HAT_FUNC = {
    SEProblemType.Logistic: logistic_var_hat_func,
    SEProblemType.LogisticFGM: logistic_fgm_var_hat_func,
}


def fixed_point_finder(
    data_model: DataModel,
    task: Task,
    log: bool = True,
) -> Overlaps:
    overlaps = Overlaps.from_se_problem_type(task.se_problem_type)

    err = 1.0
    iter_nb = 0

    while err > TOL_FPE or iter_nb < MIN_ITER_FPE:
        if iter_nb % 500 == 0 and log:
            logging.info(f"iter_nb: {iter_nb}, err: {err}")
            logging.info(f"error: {err}")

        overlaps = MAP_PROBLEM_TYPE_VAR_HAT_FUNC[task.se_problem_type](
            task, overlaps, data_model
        )

        new_overlaps = MAP_PROBLEM_TYPE_VAR_FUNC[task.se_problem_type](
            task, overlaps, data_model
        )

        err = overlaps.update_overlaps(new_overlaps)

        iter_nb += 1
        if iter_nb > MAX_ITER_FPE:
            logging.info("!!! ------------- MAX ITERATIONS REACHED -------------- !!!")
            raise Exception("fixed_point_finder - reached max_iterations")
    return overlaps
