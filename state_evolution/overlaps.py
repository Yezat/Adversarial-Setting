from typing import Iterable
from state_evolution.constants import (
    INITIAL_CONDITION,
    BLEND_FPE,
    SEProblemType,
    OVERLAPS,
    OVERLAPS_FGM,
    HAT_OVERLAPS,
    HAT_OVERLAPS_FGM,
)


def damped_update(new, old, damping) -> float:
    return damping * new + (1 - damping) * old


class Overlaps:
    def __init__(self, overlaps: Iterable[str], hat_overlaps: Iterable[str]) -> None:
        super().__setattr__("_overlaps", {k: INITIAL_CONDITION for k in overlaps})
        super().__setattr__("_hat_overlaps", {k: 0 for k in hat_overlaps})

    def __getattr__(self, name) -> float:
        if name in self._overlaps:
            return self._overlaps[name]
        if name in self._hat_overlaps:
            return self._hat_overlaps[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value) -> None:
        if name in self._overlaps:
            self._overlaps[name] = value
        elif name in self._hat_overlaps:
            self._hat_overlaps[name] = value
        else:
            raise AttributeError(f"Cannot add new attribute '{name}' dynamically.")

    def __getitem__(self, key) -> float:
        return self.__getattr__(key)

    def __setitem__(self, key, value) -> None:
        self.__setattr__(key, value)

    @classmethod
    def from_se_problem_type(self, se_problem_type: SEProblemType) -> "Overlaps":
        match se_problem_type:
            case SEProblemType.Logistic:
                return Overlaps(OVERLAPS, HAT_OVERLAPS)
            case SEProblemType.LogisticFGM:
                return Overlaps(OVERLAPS_FGM, HAT_OVERLAPS_FGM)

    def update_overlaps(self, other: "Overlaps") -> float:
        err = 0
        for overlap, value in self._overlaps.items():
            update = damped_update(other[overlap], value, BLEND_FPE)
            err = max(abs(value - update), err)
            self[overlap] = update

        return err
