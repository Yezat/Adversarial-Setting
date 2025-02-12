import json
import numpy as np
from enum import Enum
from typing import Any
from sweep.experiment import Experiment, ExperimentType
from erm.problems.problems import ProblemType


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

        return obj
