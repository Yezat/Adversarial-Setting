import numpy as np
from erm.problems.problems import ProblemType
from model.data import DataModel


class Task:  # TODO, is this a use-case for a dataclass?
    def __init__(
        self,
        id,
        method,  # TODO rename this to task type (optimal_lambda, state_evolution, optimal_epsilon, optimal_adversarial_lambda)
        problem_type: ProblemType,
        alpha,
        epsilon,
        test_against_epsilons: np.ndarray,
        lam,
        tau,
        d,
        ps,
        dp,
        data_model: DataModel,
        values: dict,  # TODO rename this, these are additional parameters we're either collecting or passing to the optimizer
        gamma_fair_error: float,
    ):
        self.id = id
        self.method = method
        self.alpha = alpha
        self.epsilon = epsilon
        self.test_against_epsilons: np.ndarray = test_against_epsilons
        self.lam = lam
        self.tau = tau
        self.d = d
        self.gamma = 1
        self.result = None
        self.ps = ps
        self.dp = dp
        self.data_model: DataModel = data_model
        self.values: dict = values
        self.gamma_fair_error: float = gamma_fair_error
        self.problem_type: ProblemType = problem_type
