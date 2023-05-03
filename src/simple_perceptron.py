import numpy as np
from numpy import ndarray

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.optimization_method import OptimizationMethod
from src.perceptron import Perceptron
from src.error import mse


class SimplePerceptron(Perceptron):
    def __init__(self, dim: int, epochs: int, cut_condition: CutCondition, activation_method: ActivationMethod,
                 optimization_method: OptimizationMethod):

        super().__init__(epochs, cut_condition, activation_method, optimization_method)

        self._weights = np.array([np.random.uniform(-1, 1) for _ in range(dim + 1)])

    def train_online(self, data: ndarray[float], expected: ndarray[float]) -> tuple[list[ndarray], list[ndarray]]:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        errors = np.zeros(expected.shape)
        error_history = []
        weight_history = [self._weights]
        # assert data and answers dimensions are correct
        assert data.shape[0] == expected.shape[0]
        assert data.shape[1] == self._weights.shape[0]

        for epoch in range(self._epochs):
            for i in range(data.shape[0]):
                h = np.dot(data[i], self._weights)
                result = self._activation_function.evaluate(h)
                derivative = self._activation_function.d_evaluate(h)
                errors[i] = expected[i] - result

                delta = errors[i] * derivative
                self._weights += self._optimization_method.adjust(delta, data[i])
                weight_history.append(np.copy(self._weights))

            error_history.append(mse(errors))
            if self._cut_condition.is_finished(errors):
                break

        return error_history, weight_history

    def train_batch(self, data: ndarray[float], expected: ndarray[float]) -> tuple[list[ndarray], list[ndarray]]:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        error_history = []
        weight_history = [self._weights]

        # assert data and answers dimensions are correct
        assert data.shape[0] == expected.shape[0]
        assert data.shape[1] == self._weights.shape[0]

        for epoch in range(self._epochs):
            h = np.dot(data, self._weights)
            results = self._activation_function.evaluate(h)
            derivatives = self._activation_function.d_evaluate(h)
            errors = expected - results
            error_history.append(mse(errors))

            if self._cut_condition.is_finished(errors):
                break

            delta = errors * derivatives
            self._weights += self._optimization_method.adjust(delta, data)
            weight_history.append(np.copy(self._weights))

        return error_history, weight_history

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        # Add a 1 for w0
        data = np.insert(np.atleast_2d(data), 0, 1, axis=1)
        h = np.dot(data, self._weights)
        return self._activation_function.evaluate(h)
