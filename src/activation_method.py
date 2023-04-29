from abc import ABC
from numbers import Number


class ActivationMethod(ABC):
    @staticmethod
    def evaluate(x: Number) -> Number:
        raise NotImplementedError()

    @staticmethod
    def d_evaluate(x: Number) -> Number:
        raise NotImplementedError()


class StepActivationFunction(ActivationMethod):
    @staticmethod
    def evaluate(x: Number) -> Number:
        return 1 if x >= 0 else -1

    @staticmethod
    def d_evaluate(x: Number) -> Number:
        return 1 if x == 0 else 0
