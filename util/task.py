import numpy as np
from enum import Enum
from typing import Any
from erm.problems.problems import ProblemType
from dataclasses import dataclass
from model.data import DataModel


class TaskType(Enum):
    ERM = 0
    SE = 1
    OL = 2


@dataclass
class Task:
    id: int
    task_type: TaskType
    erm_problem_type: ProblemType
    alpha: float
    epsilon: float
    test_against_epsilons: np.ndarray
    lam: float
    tau: float
    d: int
    data_model: DataModel
    values: dict
    gamma_fair_error: float
    gamma: float = 1
    result: Any = None
