import os.path
import statistics
from abc import ABC
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np

import utils
from src.activation_method import StepActivationFunction, IdentityActivationFunction, TangentActivationFunction, \
    SigmoidActivationFunction
from src.cut_condition import FalseCutCondition, AccuracyCutCondition, MSECutCondition
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import GradientDescentOptimization
from src.simple_perceptron import SimplePerceptron

OUTPUT_DIR = "figs/"
TEST_COUNT = 10
MAX_EPOCHS = 10000
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


class ErrorVsEpochTestPlotter(TestPlotter):
    def __init__(self, out_name, title, xaxis, yaxis):
        super().__init__(out_name)
        self._title = title
        self._xaxis = xaxis
        self._yaxis = yaxis

    def _create_data(self):
        return []

    def _add_data(self, data, new_data):
        data.append(new_data)

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

    def _save_plot(self, data):
        # plot
        fig, ax = plt.subplots()
        self._plot_line(fig, ax, data)

        plt.title(self._title)
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()

        plt.savefig(OUTPUT_DIR + self._out_name)

    def _plot_line(self, fig, ax, data, label=None):
        mean = np.array(data["mean"])
        std = np.array(data["std"])

        x = np.arange(mean.shape[0])

        ax.fill_between(x, mean + std, mean - std, alpha=.5, linewidth=0, label=label)
        ax.plot(x, mean, '.-', linewidth=2)


class MultiErrorVsEpochTestPlotter(ErrorVsEpochTestPlotter):
    def __init__(self, out_name, title, xaxis, yaxis, label_type, labels):
        super().__init__(out_name, title, xaxis, yaxis)
        self._label_type = label_type
        self._labels = labels

    def _create_data(self):
        data = {}
        for label in self._labels:
            data[label] = []
        return data

    def _add_data(self, data, new_data):
        for i, label in enumerate(self._labels):
            data[label].append(new_data[i])

    def _post_process(self, data):
        post_data = {}
        for label in self._labels:
            post_data[label] = super()._post_process(data[label])

        print(post_data)
        return post_data

    def _save_plot(self, data):
        fig, ax = plt.subplots()
        for label in self._labels:
            super()._plot_line(fig, ax, data[label], f"{self._label_type} = {label}")

        plt.title(self._title)
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()
        leg = plt.legend(loc='upper right')

        plt.savefig(OUTPUT_DIR + self._out_name)


def plots_e1():
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

    learning_rates = [10, 5, 1, 0.1, 0.05]
    MultiErrorVsEpochTestPlotter("AND_error_vs_epoch_multiLR.png",
                                 f"AND: test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(2,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   StepActivationFunction(),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected[0])[0]
                  for lr in learning_rates]
         )
    )


def plots_e2():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    path = "./data/data.csv"
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = np.array(data[:, :-1])  # All rows, all columns except the last (output)
    expected = np.array(data[:, -1])  # All rows, last column

    ErrorVsEpochTestPlotter("DATA_IDENTITY_error_vs_epoch.png",
                            f"LINEAR: Learning rate = {LEARNING_RATE}, test count = {TEST_COUNT}",
                            "Epoch",
                            "Error(MSE)"
                            ).plot(
        lambda: SimplePerceptron(X[0].size,
                                 MAX_EPOCHS,
                                 FalseCutCondition(),
                                 IdentityActivationFunction(),
                                 GradientDescentOptimization(LEARNING_RATE)
                                 ).train_batch(X, expected)[0]
    )

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    MultiErrorVsEpochTestPlotter("EX2_LINEAR_error_vs_epoch_multiLR.png",
                                 f"LINEAR: test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   IdentityActivationFunction(),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_error_vs_epoch_multiLR.png",
                                 f"TANGENT: test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   TangentActivationFunction(0.5),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )


def plots_e3a():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    X = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ])
    EXPECTED = np.array([
        [-1],
        [1],
        [1],
        [-1]
    ])
    learning_rates = [10, 1, 0.1, 0.01, 0.001]
    MultiErrorVsEpochTestPlotter("E3a_ML_XOR.png",
                                 f"MultiLayer[2, 2, 1] XOR: test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [MultiLayerPerceptron([2, 2, 1],
                                       MAX_EPOCHS,
                                       FalseCutCondition(),
                                       TangentActivationFunction(0.5),
                                       GradientDescentOptimization(lr)).train_batch(X, EXPECTED)
                  for lr in learning_rates]
         )
    )


def plots_e3b():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    X = utils.get_numbers({"path": "data/TP3-ej3-digitos.txt"})
    EXPECTED = np.array([[1 - 2 * (i % 2)] for i in range(len(X))])
    TRAIN_RATIO = 0.8
    train_index = int(TRAIN_RATIO * (X.shape[0] - 1)) + 1

    learning_rates = [1, 0.1, 0.01, 0.001]
    MultiErrorVsEpochTestPlotter("E3b_ML_Parity.png",
                                 f"MultiLayer[{X.shape[1]}, 5, {EXPECTED.shape[1]}] Parity: test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [MultiLayerPerceptron([X.shape[1], 5, EXPECTED.shape[1]],
                                       MAX_EPOCHS,
                                       FalseCutCondition(),
                                       TangentActivationFunction(0.5),
                                       GradientDescentOptimization(lr)).train_batch(
            X[:train_index, :],
            EXPECTED[:train_index, :]
        )
            for lr in learning_rates]
         )
    )


if __name__ == "__main__":
    plots_e3b()
