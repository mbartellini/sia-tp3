from numbers import Number
from typing import List
import numpy as np

from src.activation_function import ActivationFunction


class Perceptron:
    # TODO
    def __init__(self,
                 dimension: int,
                 activation_function: ActivationFunction):
        self._weights = [0] * dimension
        self._activation_function = activation_function

    def _epoch(self, learning_rate: Number, data):
        for d in data:
            p = self.predict(d.input)
            dw += learning_rate * (p - d.output) * d.input

    def train(self, learning_rate: Number, data):
        dw = 0
        while condition:
            self._epoch(learning_rate, data)

    def predict(self, input: List[Number]) -> Number:
        return self._activation_function.evaluate(
            np.dot(input, self._weights[1:]) - self._weights[0]
        )
