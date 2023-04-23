from abc import ABC
from numbers import Number


class ActivationFunction(ABC):
    def evaluate(self, x: Number) -> Number:
        raise NotImplementedError()


class StepActivationFunction(ActivationFunction):
    def evaluate(self, x: Number) -> Number:
        return 1 if x >= 0 else -1
