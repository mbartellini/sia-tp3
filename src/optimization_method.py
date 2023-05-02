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


class MomentumOptimization(OptimizationMethod):
    def __init__(self, alpha=0.3, learning_rate=0.1):
        super().__init__(learning_rate)
        self._alpha = alpha
        self._prev = 0

    def adjust(self, error: ndarray[float], derivative: ndarray[float], data: ndarray[float]) -> ndarray[float]:
        self._prev = self._learning_rate * np.dot(error * derivative, data) + self._alpha * self._prev
        return self._prev
