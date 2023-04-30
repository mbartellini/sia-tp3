from abc import ABC

import numpy as np
from numpy import ndarray


class ActivationMethod(ABC):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError()

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError()


class StepActivationFunction(ActivationMethod):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.where(x >= 0, 1, -1)

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.ones_like(x)


class IdentityActivationFunction(ActivationMethod):
    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return x

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.ones_like(x)


class TangentActivationFunction(ActivationMethod):
    def __init__(self, beta: float):
        self._beta = beta

    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return np.tanh(self._beta * x)

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return self._beta * (1 - self.evaluate(x) ** 2)


class LogisticActivationFunction(ActivationMethod):
    def __init__(self, beta: float):
        self._beta = beta

    def evaluate(self, x: ndarray[float]) -> ndarray[float]:
        return 1 / (1 + np.exp(-2 * self._beta * x))

    def d_evaluate(self, x: ndarray[float]) -> ndarray[float]:
        result = self.evaluate(x)
        return 2 * self._beta * result * (1 - result)
