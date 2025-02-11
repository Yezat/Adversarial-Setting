from enum import Enum
from erm.problems.logistic_problem import logistic_loss_gradient
from erm.problems.perturbed_logistic_problem import perturbed_logistic_loss_gradient
from erm.problems.coefficient_logistic_problem import coefficient_logistic_loss_gradient


class ProblemType(Enum):
    Logistic = 0
    PerturbedLogistic = 1
    CoefficientLogistic = 2


LOSS_GRADIENT_MAP = {
    ProblemType.Logistic: logistic_loss_gradient,
    ProblemType.PerturbedLogistic: perturbed_logistic_loss_gradient,
    ProblemType.CoefficientLogistic: coefficient_logistic_loss_gradient,
}
