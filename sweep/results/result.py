from abc import ABC
from util.task import Task
from datetime import datetime
from model.data import DataModel
import uuid


class Result(ABC):
    def __init__(self, task: Task, data_model: DataModel):
        self.id: str = str(uuid.uuid4())

        self.experiment_id: str = task.experiment_id
        self.date: datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data_model = repr(data_model)

        # store the task parameters
        self.__dict__.update(task.__dict__)
