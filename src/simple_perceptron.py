from typing import List

import numpy as np
from numpy import ndarray

from src.activation_method import ActivationMethod
from src.cut_condition import CutCondition
from src.optimization_method import OptimizationMethod


class SimplePerceptron:
    def __init__(self, dim: int, epochs: int, cut_condition: CutCondition,
                 activation_method: ActivationMethod, optimization_method: OptimizationMethod):
        self._weights = np.array([np.random.uniform(-1, 1) for _ in range(dim + 1)])
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

            if self._cut_condition.is_finished(errors):
                return epoch

            self._weights += self._optimization_method.adjust(errors, derivatives, data)

        return self._epochs

    def predict(self, data: ndarray[float]) -> ndarray[float]:
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        h = np.dot(data, self._weights)
        return self._activation_function.evaluate(h)


class MLP:
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int], epochs: int,
                 cut_condition: CutCondition, activation_function: ActivationMethod,
                 optimization_method: OptimizationMethod):
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
        # assert data and answers dimensions are correct
        assert data.shape[0] == answers.shape[0]
        assert data.shape[1] == self._weights[0].shape[0] - 1

        for epoch in range(self._epochs):
            # Add a 1 for w0
            results = np.insert(data, 0, 1, axis=1)
            feedforward_data = [results]
            feedforward_output = [data]

            # Forward pass
            for i in range(len(self._weights)):
                h = np.dot(results, self._weights[i])
                feedforward_data.append(h)
                results = self._activation_function.evaluate(h)
                feedforward_output.append(results)
                results = np.insert(results, 0, 1, axis=1)  # Add w0 for next layer

            derivatives = self._activation_function.d_evaluate(feedforward_data[-1])
            print(answers, feedforward_output[-1])
            errors = answers - feedforward_output[-1]
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
