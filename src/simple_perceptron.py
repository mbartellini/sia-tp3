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
            for i in range(data.shape[0]):
                h = np.dot(data[i], self._weights)
                result = self._activation_function.evaluate(h)
                derivative = self._activation_function.d_evaluate(h)
                errors[i] = answers[i] - result
                self._weights += self._optimization_method.adjust(errors[i], derivative, data[i])

            if self._cut_condition.is_finished(errors):
                return epoch

        return self._epochs

    def train_batch(self, data: ndarray[float], answers: ndarray[float]) -> int:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)

        # assert data and answers dimensions are correct
        assert data.shape[0] == answers.shape[0]
        assert data.shape[1] == self._weights.shape[0]

        for epoch in range(self._epochs):
            h = np.dot(data, self._weights)
            results = self._activation_function.evaluate(h)
            derivatives = self._activation_function.d_evaluate(h)
            errors = answers - results
            self._weights += self._optimization_method.adjust(errors, derivatives, data)

            if self._cut_condition.is_finished(errors):
                return epoch

        return self._epochs

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        h = np.dot(data, self._weights)
        return self._activation_function.evaluate(h)


class MLP:
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int],
                 activation_function: ActivationMethod, optimization_method: OptimizationMethod,
                 cut_condition: CutCondition, epochs: int):
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_layers = hidden_layers
        self._activation_function = activation_function
        self._optimization_method = optimization_method
        self._cut_condition = cut_condition
        self._epochs = epochs

        # Initialize weights with random values between -1 and 1
        sizes = [input_size] + hidden_layers + [output_size]
        self._weights = [np.random.uniform(-1, 1, (sizes[i] + 1, sizes[i + 1])) for i in range(len(sizes) - 1)]

    def train_batch(self, data: ndarray[float], answers: ndarray[float]) -> int:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)

        # assert data and answers dimensions are correct
        assert data.shape[0] == answers.shape[0]
        assert data.shape[1] == self._weights[0].shape[0]

        for epoch in range(self._epochs):
            activations = [data]
            h = data

            # Forward pass
            for i in range(len(self._weights)):
                h = np.dot(h, self._weights[i])
                results = self._activation_function.evaluate(h)
                activations.append(results)
                h = np.insert(results, 0, 1, axis=1)  # Add bias for next layer

            derivatives = self._activation_function.d_evaluate(h)
            errors = answers - activations[-1]
            deltas = [errors * derivatives]

            # Backward pass
            for i in range(len(self._weights) - 1, 0, -1):
                delta = np.dot(deltas[-1], self._weights[i].T)
                delta = delta[:, 1:]  # Remove bias
                derivative = self._activation_function.d_evaluate(activations[i])
                delta *= derivative
                deltas.append(delta)

            deltas.reverse()

            # Update weights
            for i in range(len(self._weights)):
                self._weights[i] += self._optimization_method.adjust(deltas[i], activations[i], data)

            if self._cut_condition.is_finished(errors):
                return epoch

        return self._epochs
