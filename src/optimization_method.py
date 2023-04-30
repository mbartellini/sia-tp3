from abc import ABC

from numpy import ndarray


class OptimizationMethod(ABC):
    def __init__(self, learning_rate=0.1):
        self._learning_rate = learning_rate

    def adjust(self, error, derivative, row):
        raise NotImplementedError()


class GradientDescentOptimization(OptimizationMethod):
    def adjust(self, error: float, derivative: ndarray[float], row: ndarray[float]) -> ndarray[float]:
        return self._learning_rate * error * derivative * row
