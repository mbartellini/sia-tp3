import os.path
import statistics
import tracemalloc
import time
from abc import ABC
from numbers import Number
from typing import List, Dict, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from src.activation_method import StepActivationFunction
from src.cut_condition import AccuracyCutCondition, FalseCutCondition
from src.optimization_method import GradientDescentOptimization
from src.simple_perceptron import SimplePerceptron

OUTPUT_DIR = "figs/"
TEST_COUNT = 1
MAX_EPOCHS = 50
LEARNING_RATE = 0.01


class TestPlotter(ABC):

    def __init__(self, out_name: str):
        self._out_name = out_name

    def plot(self, test: Callable):
        data = self._create_data()

        for t in range(TEST_COUNT):
            self._add_data(data, test())

        data = self._post_process(data)

        self._save_plot(data)

    def _create_data(self):
        raise NotImplementedError()

    def _add_data(self, data, new_data):
        raise NotImplementedError

    def _post_process(self, data):
        raise NotImplementedError()

    def _save_plot(self, data):
        raise NotImplementedError()


class AveragePostProcessMixin:
    def _post_process(self, data):
        max_size = max([len(history) for history in data])
        mean, std = [], []
        for i in range(max_size):
            data_of_current_epoch = list(filter(
                lambda x: x is not None,
                [history[i] if i < len(history) else None for history in data]
            ))
            mean.append(statistics.mean(data_of_current_epoch))
            if len(data_of_current_epoch) >= 2:
                std.append(statistics.stdev(data_of_current_epoch))
            else:
                std.append(.0)

        return {
            "mean": mean,
            "std": std,
        }


class ErrorVsEpochTestPlotter(AveragePostProcessMixin, TestPlotter):
    def __init__(self, out_name, title, xaxis, yaxis):
        super().__init__(out_name)
        self._title = title
        self._xaxis = xaxis
        self._yaxis = yaxis

    def _create_data(self):
        return []

    def _add_data(self, data, new_data):
        data.append(new_data)

    def _save_plot(self, data):
        mean = np.array(data["mean"])
        std = np.array(data["std"])

        x = np.arange(mean.shape[0])
        # plot
        fig, ax = plt.subplots()

        ax.fill_between(x, mean+std, mean-std, alpha=.5, linewidth=0)
        ax.plot(x, mean, 'o-', linewidth=2)

        plt.title(self._title)
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()

        plt.savefig(OUTPUT_DIR + self._out_name)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    expected = np.array([[-1, -1, -1, 1], [-1, 1, 1, -1]])

    ErrorVsEpochTestPlotter("AND_error_vs_epoch.png",
                            f"AND: Learning rate = {LEARNING_RATE}, test count = {TEST_COUNT}",
                            "Epoch",
                            "Error(MSE)"
                            ).plot(
        lambda: SimplePerceptron(2,
                                 MAX_EPOCHS,
                                 FalseCutCondition(),
                                 StepActivationFunction(),
                                 GradientDescentOptimization(LEARNING_RATE)
                                 ).train_batch(X, expected[0])[0]
    )

    ErrorVsEpochTestPlotter("XOR_error_vs_epoch.png",
                            f"XOR: Learning rate = {LEARNING_RATE}, test count = {TEST_COUNT}",
                            "Epoch",
                            "Error(MSE)"
                            ).plot(
        lambda: SimplePerceptron(2,
                                 MAX_EPOCHS,
                                 FalseCutCondition(),
                                 StepActivationFunction(),
                                 GradientDescentOptimization(LEARNING_RATE)
                                 ).train_batch(X, expected[1])[0]
    )

