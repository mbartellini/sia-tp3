import math
from abc import ABC


class ActivationMethod(ABC):
    def evaluate(self, x: float) -> float:
        raise NotImplementedError()

    def d_evaluate(self, x: float) -> float:
        raise NotImplementedError()


class StepActivationFunction(ActivationMethod):
    def evaluate(self, x: float) -> float:
        return 1 if x >= 0 else -1

    def d_evaluate(self, x: float) -> float:
        return 1


class IdentityActivationFunction(ActivationMethod):
    def evaluate(self, x: float) -> float:
        return x

    def d_evaluate(self, x: float) -> float:
        return 1


class TangentActivationFunction(ActivationMethod):
    def __init__(self, beta: float):
        self._beta = beta

    def evaluate(self, x: float) -> float:
        return math.tanh(self._beta * x)

    def d_evaluate(self, x: float) -> float:
        return self._beta * (1 - self.evaluate(x) ** 2)


class LogisticActivationFunction(ActivationMethod):
    def __init__(self, beta: float):
        self._beta = beta

    def evaluate(self, x: float) -> float:
        return 1 / (1 + math.exp(-2 * self._beta * x))

    def d_evaluate(self, x: float) -> float:
        return 2 * self._beta * self.evaluate(x) * (1 - self.evaluate(x))
