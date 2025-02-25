from enum import Enum
from model.data import DataModel, KFeaturesDefinition
from erm.problems.problems import ProblemType
from util.task import Task, TaskType
from typing import Iterable, Iterator, Any
import numpy as np
from itertools import product
import json
from copy import deepcopy


class ExperimentType(Enum):
    Sweep = 0
    OptimalLambda = 1


MAP_EXPERIMENT_TYPE_TO_TASK_TYPE_SE = {
    ExperimentType.Sweep: TaskType.SE,
    ExperimentType.OptimalLambda: TaskType.OL,
}


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
        data_models: list[DataModel],
        test_against_epsilons: Iterable[float],
        erm_problem_type: ProblemType,
        gamma_fair_error: float,
        name: str = "",
    ) -> None:
        self.name: str = name
        self.state_evolution_repetitions: int = state_evolution_repetitions
        self.erm_repetitions: int = erm_repetitions
        self.alphas: Iterable[float] = alphas
        self.epsilons: Iterable[float] = epsilons
        self.test_against_epsilons: Iterable[float] = test_against_epsilons
        self.lambdas: Iterable[float] = lambdas
        self.taus: Iterable[float] = taus
        self.d: int = d
        self.experiment_type: ExperimentType = experiment_type
        self.data_models: list[DataModel] = data_models
        self.erm_problem_type: ProblemType = erm_problem_type
        self.gamma_fair_error: float = gamma_fair_error

    @classmethod
    def fromdict(cls, d) -> "Experiment":
        d["data_models"] = [DataModel.from_dict(m) for m in d["data_models"]]

        d["experiment_type"] = ExperimentType[d["experiment_type"]]

        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, cls=NumpyEncoder)

    def __iter__(self) -> Iterator[Task]:
        attributes = {
            "alpha": self.alphas,
            "epsilon": self.epsilons,
            "lam": self.lambdas,
            "tau": self.taus,
            "data_model": self.data_models,
        }

        task_id = 0

        def _get_task(
            combination: dict,
            task_id: int,
            task_type: TaskType,
            erm_problem_type: ProblemType = None,
        ) -> Task:
            return Task(
                id=task_id,
                task_type=task_type,
                erm_problem_type=erm_problem_type,
                alpha=combination["alpha"],
                epsilon=combination["epsilon"],
                test_against_epsilons=self.test_against_epsilons,
                lam=combination["lam"],
                tau=combination["tau"],
                d=self.d,
                data_model=combination["data_model"],
                values={},
                gamma_fair_error=self.gamma_fair_error,
            )

        for _ in range(self.state_evolution_repetitions):
            for combination in product(*attributes.values()):
                comb = dict(zip(attributes.keys(), combination))
                task_id += 1
                yield _get_task(
                    comb,
                    task_id,
                    MAP_EXPERIMENT_TYPE_TO_TASK_TYPE_SE[self.experiment_type],
                )

        for _ in range(self.erm_repetitions):
            for combination in product(*attributes.values()):
                comb = dict(zip(attributes.keys(), combination))
                task_id += 1
                yield _get_task(
                    comb, task_id, TaskType.ERM, erm_problem_type=ProblemType.Logistic
                )


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Experiment):
            return obj.__dict__
        if isinstance(obj, np.int32):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, DataModel):
            dict_copy = deepcopy(obj.__dict__)
            dict_copy.pop("data_model_factory")
            return dict_copy
        if isinstance(obj, KFeaturesDefinition):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def decode(self, s, _w=json.decoder.WHITESPACE.match) -> Any:
        # Parse the JSON string into a Python object
        obj = super().decode(s, _w)

        match obj:
            case "experiment_type":
                obj["experiment_type"] = ExperimentType[obj["experiment_type"]]
            case "erm_problem_type":
                obj["erm_problem_type"] = ProblemType[obj["erm_problem_type"]]
            case "data_models":
                obj["data_models"] = [
                    DataModel.from_dict(m) for m in obj["data_models"]
                ]

        return obj
