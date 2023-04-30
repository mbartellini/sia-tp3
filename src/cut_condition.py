import math
from abc import ABC


class CutCondition(ABC):
    def process_prediction(self, error: float):
        raise NotImplementedError()

    def is_finished(self) -> bool:
        raise NotImplementedError()


class AccuracyCutCondition(CutCondition):
    def __init__(self, total_predictions: int):
        self._counter = 0
        self._correct_predictions = 0

    def process_prediction(self, error: float):
        self._counter += 1
        if error == 0:
            self._correct_predictions += 1

    def is_finished(self) -> bool:
        result = self._counter == self._correct_predictions
        self._correct_predictions = 0
        self._counter = 0

        return result


class AbsoluteValueCutCondition(CutCondition):
    def __init__(self):
        self._errors = 0

    def process_prediction(self, error: float):
        self._errors += math.fabs(error)

    def is_finished(self) -> bool:
        result = self._errors == 0
        self._errors = 0

        return result


class MSECutCondition(CutCondition):
    def __init__(self, eps: float = 0.01):
        self._counter = 0
        self._errors = 0
        self._eps = eps

    def process_prediction(self, error: float):
        self._errors += error ** 2
        self._counter += 1

    def is_finished(self) -> bool:
        result = self._errors / self._counter
        self._errors = 0
        self._counter = 0

        return result < self._eps
