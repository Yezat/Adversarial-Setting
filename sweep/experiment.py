from enum import Enum
from datetime import datetime
from model.data import DataModel
from erm.problems.problems import ProblemType
from util.task import Task, TaskType
from typing import Iterable, Iterator
import uuid
from itertools import product


class ExperimentType(Enum):  # TODO clean the experiment types
    Sweep = 0
    OptimalLambda = 1
    SweepAtOptimalLambda = 2
    OptimalEpsilon = 3
    OptimalLambdaAdversarialTestError = 4
    SweepAtOptimalLambdaAdversarialTestError = 5
    OptimalAdversarialErrorEpsilon = 6
    SweepAtOptimalAdversarialErrorEpsilon = 7


class Experiment:
    def __init__(
        self,
        state_evolution_repetitions: int,
        erm_repetitions: int,
        alphas: Iterable[float],
        epsilons: Iterable[float],
        lambdas: Iterable[float],
        taus: Iterable[float],
        d: int,
        experiment_type: ExperimentType,
        data_model: DataModel,
        test_against_epsilons: Iterable[float],
        erm_problem_type: ProblemType,
        gamma_fair_error: float,
        experiment_name: str = "",
    ) -> None:
        self.experiment_id: str = str(uuid.uuid4())
        self.experiment_name: str = experiment_name
        self.date: datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state_evolution_repetitions: int = state_evolution_repetitions
        self.erm_repetitions: int = erm_repetitions
        self.alphas: Iterable[float] = alphas
        self.epsilons: Iterable[float] = epsilons
        self.test_against_epsilons: Iterable[float] = test_against_epsilons
        self.lambdas: Iterable[float] = lambdas
        self.taus: Iterable[float] = taus
        self.d: int = d
        self.experiment_type: ExperimentType = experiment_type
        self.completed: bool = False
        self.data_model: DataModel = repr(data_model)
        self.erm_problem_type: ProblemType = erm_problem_type
        self.gamma_fair_error: float = gamma_fair_error

    @classmethod
    def fromdict(cls, d) -> "Experiment":
        return cls(**d)

    def __iter__(self) -> Iterator[Task]:
        attributes = {
            "alpha": self.alphas,
            "epsilon": self.epsilons,
            "lam": self.lambdas,
            "tau": self.taus,
        }

        task_id = 0

        def _get_task(combination: dict, task_id: int, task_type: TaskType) -> Task:
            return Task(
                id=task_id,
                task_type=task_type,
                problem_type=None,
                alpha=combination["alpha"],
                epsilon=combination["epsilon"],
                test_against_epsilons=self.test_against_epsilons,
                lam=combination["lam"],
                tau=combination["tau"],
                d=self.d,
                data_model=self.data_model,
                values={},
                gamma_fair_error=self.gamma_fair_error,
            )

        for _ in range(self.state_evolution_repetitions):
            for combination in product(*attributes.values()):
                comb = dict(zip(attributes.keys(), combination))
                task_id += 1
                yield _get_task(comb, task_id, TaskType.SE)

        for _ in range(self.erm_repetitions):
            for combination in product(*attributes.values()):
                comb = dict(zip(attributes.keys(), combination))
                task_id += 1
                yield _get_task(comb, task_id, TaskType.ERM)
