from util.task import Task
from model.data import DataModel
from state_evolution.overlaps import Overlaps
from sweep.results.state_evolution import SEResult


class OptimalLambdaResult(SEResult):
    def __init__(
        self,
        task: Task,
        overlaps: Overlaps,
        data_model: DataModel,
        optimal_lambda: float,
    ) -> None:
        super().__init__(task=task, overlaps=overlaps, data_model=data_model)
        self.optimal_lambda = optimal_lambda
