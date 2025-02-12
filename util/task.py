import numpy as np
from enum import Enum
from erm.problems.problems import ProblemType
from model.data import DataModel


class TaskType(Enum):
    ERM = 0
    SE = 1


class Task:
    def __init__(
        self,
        id,
        task_type: TaskType,  # TODO rename this to task type (optimal_lambda, state_evolution, optimal_epsilon, optimal_adversarial_lambda)
        erm_problem_type: ProblemType,
        alpha,
        epsilon,
        test_against_epsilons: np.ndarray,
        lam,
        tau,
        d,
        data_model: DataModel,
        values: dict,  # TODO rename this, these are additional parameters we're either collecting or passing to the optimizer
        gamma_fair_error: float,
    ) -> None:
        self.id = id
        self.task_type: TaskType = task_type
        self.erm_problem_type: ProblemType = erm_problem_type
        self.alpha = alpha
        self.epsilon = epsilon
        self.test_against_epsilons: np.ndarray = test_against_epsilons
        self.lam = lam
        self.tau = tau
        self.d = d
        self.gamma = 1
        self.result = None
        self.data_model: DataModel = data_model
        self.values: dict = values
        self.gamma_fair_error: float = gamma_fair_error
