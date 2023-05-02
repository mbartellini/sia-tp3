from abc import ABC
from numbers import Number
from typing import List
import numpy as np

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.error_method import ErrorMethod
from src.layer import Layer
from src.optimization_method import OptimizationMethod
from src.update_method import UpdateMethod


class Perceptron(ABC):
    def __init__(self,
                 activation_method: ActivationMethod,
                 error_method: ErrorMethod,
                 learn_rate: float,
                 epochs: int,
                 update_method: UpdateMethod,
                 architecture: List[int],
                 optimization_method: OptimizationMethod,
                 cut_condition: CutCondition
                 ):

        # Initialize weights for the whole network with random [1-0] values.
        self._layers = []
        for i in range(len(architecture) - 1):
            self._layers.append(Layer(np.random.uniform(-1, 1, (architecture[i] + 1, architecture[i+1])))) 
            
        self._update_method = update_method
        self._error_method = error_method
        self._learn_rate = learn_rate
        self._activation_function = activation_method
        self._epochs = epochs
        self._cut_condition = cut_condition
        self._activation_function = activation_method
        self._optimization_method = optimization_method

    # def predict(self, stimuli: List[Number]) -> Number:
    #     return self._activation_function.evaluate(
    #         np.dot([-1]+stimuli, self._weights)
    #     )
    #
    # def train(self,
    #           learning_rate: Number,
    #           epoch_limit: int,
    #           update_method: UpdateMethod,
    #           cut_condition: CutCondition,
    #           data):
    #     for _ in range(epoch_limit):
    #         self._epoch(learning_rate, data)
    #         if cut_condition.is_finished():
    #             # TODO check cost or error function and break
    #
    # def _epoch(self,
    #            learning_rate: Number,
    #            data):
    #     for d in data:
    #         p = self.predict(d.stimuli)
    #         dw = learning_rate * (d.output - p) * np.asarray(d.stimuli)
