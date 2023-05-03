from abc import ABC

from numpy import ndarray

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.optimization_method import OptimizationMethod


class Perceptron(ABC):
    def __init__(self, epochs: int, cut_condition: CutCondition,
                 activation_method: ActivationMethod, optimization_method: OptimizationMethod):
        self._epochs = epochs
        self._cut_condition = cut_condition
        self._activation_function = activation_method
        self._optimization_method = optimization_method

    def train_batch(self, data: ndarray[float], expected: ndarray[float]):
        raise NotImplementedError

    def train_online(self, data: ndarray[float], expected: ndarray[float]):
        raise NotImplementedError

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError
