import random

import numpy as np
from numpy import ndarray

from src.cut_condition import CutCondition
from src.update_method import UpdateMethod


class SimplePerceptron:
    def __init__(self, dim: int, update_method: UpdateMethod, cut_condition: CutCondition, learning_rate: float = 0.1,
                 periods: int = 1000):
        self._weights = np.array([random.uniform(-1, 1) for _ in range(dim + 1)])
        self._learning_rate = learning_rate
        self._periods = periods
        self._update_method = update_method
        self._cut_condition = cut_condition

    def train(self, data: ndarray[float], answers: ndarray[float]):
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)

        # assert data and answers dimensions are correct
        assert data.shape[0] == answers.shape[0]
        assert data.shape[1] == self._weights.shape[0]

        for _ in range(self._periods):
            for i in range(data.shape[0]):
                row = data[i]
                h = np.dot(row, self._weights)
                if h >= 0:
                    result = 1
                else:
                    result = -1

                error = answers[i] - result
                self._cut_condition.process_prediction(error)
                dw = self._learning_rate * error * row
                self._weights = self._update_method.process_prediction(self._weights, dw)

            self._update_method.process_epoch(self._weights)

            if self._cut_condition.is_finished():
                break

    def predict(self, data: ndarray[float]):
        # Add a 1 for w0
        data = np.insert(data, 0, 1, axis=1)
        rows_count = data.shape[0]
        answers = np.zeros(rows_count)
        for i in range(rows_count):
            if np.dot(data[i], self._weights) >= 0:
                answers[i] = 1
            else:
                answers[i] = -1
        return answers
