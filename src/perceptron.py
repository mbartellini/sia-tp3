from abc import ABC
from typing import List

import numpy as np
from numpy import ndarray

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.error_method import ErrorMethod
from src.layer import Layer
from src.optimization_method import OptimizationMethod


class Perceptron(ABC):
    def __init__(self,
                 activation_method: ActivationMethod,
                 error_method: ErrorMethod,
                 learn_rate: float,
                 epochs: int,
                 architecture: List[int],
                 optimization_method: OptimizationMethod,
                 cut_condition: CutCondition
                 ):

        # Initialize weights for the whole network with random [1-0] values.
        self._layers = []
        for i in range(len(architecture) - 1):
            self._layers.append(Layer(np.random.uniform(-1, 1, (architecture[i] + 1, architecture[i+1])))) 
            
        self._error_method = error_method
        self._learn_rate = learn_rate
        self._activation_function = activation_method
        self._epochs = epochs
        self._cut_condition = cut_condition
        self._activation_function = activation_method
        self._optimization_method = optimization_method

    def test(self, data: ndarray[float]) -> ndarray[float]:
        raise NotImplementedError

    def train_batch(self, initial_data: ndarray[float], expected: ndarray[float]):
        raise NotImplementedError
