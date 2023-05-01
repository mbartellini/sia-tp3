from abc import ABC

import numpy as np
from numpy import ndarray


class OptimizationMethod(ABC):
    def __init__(self, learning_rate=0.1):
        self._learning_rate = learning_rate

    def adjust(self, error, derivative, data):
        raise NotImplementedError()


class GradientDescentOptimization(OptimizationMethod):
    def adjust(self, error: ndarray[float], derivative: ndarray[float], data: ndarray[float]) -> ndarray[float]:
        return self._learning_rate * np.dot(error * derivative, data)
