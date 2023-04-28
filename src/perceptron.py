from numbers import Number
from typing import List
import numpy as np

from src.activation_function import ActivationFunction
from src.update_method import UpdateMethod


class Perceptron:
    # TODO
    def __init__(self,
                 dimension: int,
                 activation_function: ActivationFunction):
        self._weights = np.zeros(dimension) # TODO: Check +1 for w_0
        self._activation_function = activation_function

    def predict(self, stimuli: List[Number]) -> Number:
        return self._activation_function.evaluate(
            np.dot([-1]+stimuli, self._weights)
        )

    def train(self,
              learning_rate: Number,
              epoch_limit: int,
              update_method: UpdateMethod,
              cut_condition: CutCondition,
              data):
        for _ in range(epoch_limit):
            self._epoch(learning_rate, data)
            if cut_condition.is_finished():
                # TODO check cost or error function and break

    def _epoch(self,
               learning_rate: Number,
               data):
        for d in data:
            p = self.predict(d.stimuli)
            dw = learning_rate * (d.output - p) * np.asarray(d.stimuli)
