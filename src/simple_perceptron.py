import random

import numpy as np
from numpy import ndarray

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.optimization_method import OptimizationMethod


class SimplePerceptron:
    def __init__(self, dim: int, epochs: int, cut_condition: CutCondition,
                 activation_method: ActivationMethod, optimization_method: OptimizationMethod):
        self._weights = np.array([random.uniform(-1, 1) for _ in range(dim + 1)])
        self._epochs = epochs
        self._cut_condition = cut_condition
        self._activation_function = activation_method
        self._optimization_method = optimization_method

    def train_online(self, data: ndarray[float], answers: ndarray[float]) -> int:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        errors = np.zeros(answers.shape)
        # assert data and answers dimensions are correct
        assert data.shape[0] == answers.shape[0]
        assert data.shape[1] == self._weights.shape[0]

        for epoch in range(self._epochs):
            # TODO remove this for as it is not efficient
            for i in range(data.shape[0]):
                row = data[i]
                h = np.dot(row, self._weights)
                result = self._activation_function.evaluate(h)
                derivative = self._activation_function.d_evaluate(h)
                errors[i] = answers[i] - result
                dw = self._optimization_method.adjust(errors[i], derivative, row)
                self._weights += dw

            if self._cut_condition.is_finished(errors):
                return epoch

        return self._epochs

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        h = np.dot(data, self._weights)
        return self._activation_function.evaluate(h)
