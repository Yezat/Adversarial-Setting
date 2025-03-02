from enum import Enum
from erm.problems.logistic_problem import logistic_loss_gradient
from erm.problems.logistic_fgm_problem import logistic_fgm_loss_gradient
from erm.problems.perturbed_logistic_problem import perturbed_logistic_loss_gradient
from erm.problems.coefficient_logistic_problem import coefficient_logistic_loss_gradient
from erm.errors import (
    compute_boundary_loss,
    compute_boundary_loss_fgm,
    adversarial_error,
    adversarial_error_fgm,
)


class ProblemType(Enum):
    Logistic = 0
    PerturbedLogistic = 1
    CoefficientLogistic = 2
    LogisticFGM = 3


LOSS_GRADIENT_MAP = {
    ProblemType.Logistic: logistic_loss_gradient,
    ProblemType.PerturbedLogistic: perturbed_logistic_loss_gradient,
    ProblemType.CoefficientLogistic: coefficient_logistic_loss_gradient,
    ProblemType.LogisticFGM: logistic_fgm_loss_gradient,
}

BOUNDARY_LOSS_MAP = {
    ProblemType.Logistic: compute_boundary_loss,
    ProblemType.PerturbedLogistic: compute_boundary_loss,
    ProblemType.CoefficientLogistic: compute_boundary_loss,
    ProblemType.LogisticFGM: compute_boundary_loss_fgm,
}

ADVERSARIAL_ERROR_MAP = {
    ProblemType.Logistic: adversarial_error,
    ProblemType.PerturbedLogistic: adversarial_error,
    ProblemType.CoefficientLogistic: adversarial_error,
    ProblemType.LogisticFGM: adversarial_error_fgm,
}
