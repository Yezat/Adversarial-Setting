from abc import ABC

from util.task import Task
from datetime import datetime
import uuid


class Result(ABC):
    def __init__(self, task: Task) -> None:
        self.id: str = str(uuid.uuid4())

        self.date: datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # store the task parameters
        self.task_type = str(task.task_type)
        self.erm_problem_type = str(task.erm_problem_type)
        self.alpha = task.alpha
        self.epsilon = task.epsilon
        self.test_against_epsilons = task.test_against_epsilons
        self.lam = task.lam
        self.tau = task.tau
        self.d = task.d
        self.data_model_name = task.data_model.name
        self.values = task.values
        self.gamma_fair_error = task.gamma_fair_error
        self.gamma = task.gamma
