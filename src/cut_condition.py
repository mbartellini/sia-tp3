from abc import ABC


class CutCondition(ABC):
    def process_prediction(self, error: float):
        raise NotImplementedError()

    def is_finished(self) -> bool:
        raise NotImplementedError()


class AccuracyCutCondition(CutCondition):
    def __init__(self, total_predictions: int):
        self._total_predictions = total_predictions
        self._correct_predictions = 0

    def process_prediction(self, error: float):
        if error == 0:
            self._correct_predictions += 1

    def is_finished(self) -> bool:
        result = self._total_predictions == self._correct_predictions
        self._correct_predictions = 0

        return result


class AbsoluteValueCutCondition(CutCondition):
    def __init__(self):
        self._errors = 0

    def process_prediction(self, error: float):
        self._errors += error

    def is_finished(self) -> bool:
        result = self._errors == 0
        self._errors = 0

        return result
