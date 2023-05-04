import os.path
import statistics
from abc import ABC
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import utils
from src.activation_method import StepActivationFunction, IdentityActivationFunction, TangentActivationFunction, \
    SigmoidActivationFunction, LogisticActivationFunction
from src.cut_condition import FalseCutCondition
from src.error import mse
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import GradientDescentOptimization, MomentumOptimization
from src.simple_perceptron import SimplePerceptron

OUTPUT_DIR = "figs/"
TEST_COUNT = 100
MAX_EPOCHS = 1000


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

        plt.title(self._title, ontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.xlabel(self._xaxis)
        plt.ylabel(self._yaxis)
        plt.grid()

        plt.savefig(OUTPUT_DIR + self._out_name)

    def _plot_line(self, fig, ax, data, label=None):
        mean = np.array(data["mean"])
        std = np.array(data["std"])

        x = np.arange(mean.shape[0])

        ax.fill_between(x, mean + std, mean - std, alpha=.5, linewidth=0, label=label)
        ax.plot(x, mean, linewidth=2)


class ErrorVsTrainRatioTestPlotter:
    def __init__(self, out_name, title, xaxis, yaxis):
        self._out_name = out_name
        self._title = title
        self._xaxis = xaxis
        self._yaxis = yaxis

    def plot(self, test: Callable):
        data = []

        for i, ratio in enumerate(np.arange(0, 1, 0.05)):
            data.append([])
            for t in range(TEST_COUNT):
                data[i].append(test(ratio))

        data = self._post_process(data)

        self._save_plot(data)

    def _post_process(self, data):
        mean, std = [], []
        for data_of_current_ratio in data:
            mean.append(statistics.mean(data_of_current_ratio))
            if len(data_of_current_ratio) >= 2:
                std.append(statistics.stdev(data_of_current_ratio))
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

        x = np.arange(0, 1, 0.05)

        ax.fill_between(x, mean + std, mean - std, alpha=.5, linewidth=0, label=label)
        ax.plot(x, mean, 'o-', linewidth=2)


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

    # ErrorVsEpochTestPlotter("DATA_IDENTITY_error_vs_epoch.png",
    #                         f"LINEAR: Learning rate = {LEARNING_RATE}, test count = {TEST_COUNT}",
    #                         "Epoch",
    #                         "Error(MSE)"
    #                         ).plot(
    #     lambda: SimplePerceptron(X[0].size,
    #                              MAX_EPOCHS,
    #                              FalseCutCondition(),
    #                              IdentityActivationFunction(),
    #                              GradientDescentOptimization(LEARNING_RATE)
    #                              ).train_batch(X, expected)[0]
    # )

    learning_rates = [0.001, 0.0005, 0.0001]
    # MultiErrorVsEpochTestPlotter("EX2_LINEAR_MOMENTUM_error_vs_epoch_multiLR.png",
    #                              f"Lineal ; Activacion: Identidad ; OPT: Momentum \n alpha = 0.5 ; test count = {TEST_COUNT}",
    #                              "Epoch",
    #                              "Error(MSE)",
    #                              "LR",
    #                              learning_rates
    #                              ).plot(
    #     (lambda: [SimplePerceptron(X[0].size,
    #                                MAX_EPOCHS,
    #                                FalseCutCondition(),
    #                                IdentityActivationFunction(),
    #                                MomentumOptimization(alpha=0.5, learning_rate=lr)
    #                                ).train_batch(X, expected)[0]
    #               for lr in learning_rates]
    #      )
    # )

    # # learning_rates = [0.1, 0.001, 0.0005]
    # MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_MOMENTUM_error_vs_epoch_multiLR.png",
    #                              f"No lineal ; Activacion: Tangente(0.5) ; OPT: Momentum \n alpha = 0.5 ; test count = {TEST_COUNT}",
    #                              "Epoch",
    #                              "Error(MSE)",
    #                              "LR",
    #                              learning_rates
    #                              ).plot(
    #     (lambda: [SimplePerceptron(X[0].size,
    #                                MAX_EPOCHS,
    #                                FalseCutCondition(),
    #                                TangentActivationFunction(0.5),
    #                                MomentumOptimization(alpha=0.5, learning_rate=lr)
    #                                ).train_batch(X, expected)[0]
    #               for lr in learning_rates]
    #      )
    # )

    # MultiErrorVsEpochTestPlotter("EX2_LINEAR_SGD_error_vs_epoch_multiLR.png",
    #                              f"Lineal ; Activacion: Identidad: ; OPT: SGD ; test count = {TEST_COUNT}",
    #                              "Epoch",
    #                              "Error(MSE)",
    #                              "LR",
    #                              learning_rates
    #                              ).plot(
    #     (lambda: [SimplePerceptron(X[0].size,
    #                                MAX_EPOCHS,
    #                                FalseCutCondition(),
    #                                IdentityActivationFunction(),
    #                                GradientDescentOptimization(lr)
    #                                ).train_batch(X, expected)[0]
    #               for lr in learning_rates]
    #      )
    # )

    # learning_rates = [0.1, 0.001, 0.0005]
    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_TANGENT_0.5__error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Tangente(0.5) ; OPT: SGD ; test count = {TEST_COUNT}",
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

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_TANGENT_0.3_error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Tangente(0.3) ; OPT: SGD ; test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   TangentActivationFunction(0.3),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_TANGENT_0.7_error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Tangente(0.7) ; OPT: SGD ; test count = {TEST_COUNT}",
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

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_SIGMOID_error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Sigmoid ; OPT: SGD ; test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   SigmoidActivationFunction(),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_LOGISTIC_0.5__error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Logistic(0.5) ; OPT: SGD ; test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   LogisticActivationFunction(0.5),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_LOGISTIC_0.3_error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Logistic(0.3) ; OPT: SGD ; test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   LogisticActivationFunction(0.3),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )

    MultiErrorVsEpochTestPlotter("EX2_NON_LINEAR_SGD_LOGISTIC_0.7_error_vs_epoch_multiLR.png",
                                 f"No lineal ; Activacion: Logistic(0.7) ; OPT: SGD ; test count = {TEST_COUNT}",
                                 "Epoch",
                                 "Error(MSE)",
                                 "LR",
                                 learning_rates
                                 ).plot(
        (lambda: [SimplePerceptron(X[0].size,
                                   MAX_EPOCHS,
                                   FalseCutCondition(),
                                   LogisticActivationFunction(0.7),
                                   GradientDescentOptimization(lr)
                                   ).train_batch(X, expected)[0]
                  for lr in learning_rates]
         )
    )


def big_function(data, ratio):
    np.random.shuffle(data)
    X = np.array(data[:, :-1])  # All rows, all columns except the last (output)
    expected = np.array(data[:, -1])  # All rows, last column

    activation_function = LogisticActivationFunction(0.5)

    perceptron = SimplePerceptron(X.shape[1],
                                  MAX_EPOCHS,
                                  FalseCutCondition(),
                                  activation_function,
                                  MomentumOptimization()
                                  )

    expected = utils.scale(expected, activation_function.limits())

    train_index = int(ratio * (X.shape[0] - 1)) + 1

    train = X[:train_index, :]
    train_expected = expected[:train_index]
    test = X[train_index:, :]
    test_expected = expected[train_index:]

    perceptron.train_batch(train, train_expected)

    ans = perceptron.predict(test)

    return mse(ans - test_expected)


def plots_e2_generalization():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    path = "./data/data.csv"
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    ErrorVsTrainRatioTestPlotter("generalization_uniform.png",
                                 f"Uniform Partitioning: Activacion: Logistic(0.5) \n"
                                 f"OPT: Momentum; Learning rate = {LEARNING_RATE}; test count = {TEST_COUNT}",
                                 "Train Ratio",
                                 "Error(MSE)"
                                 ).plot(lambda ratio: (big_function(data, ratio)))


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
         ))


def plots_e3c():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    X = utils.get_numbers({"path": "data/TP3-ej3-digitos.txt"})
    y = np.array([[int(i == j) for j in range(10)] for i in range(len(X))])
    noises = [0.1 * i for i in range(10)]
    testing_size = 1000

    # learning_rates = [0.1, 0.01, 0.001, 0.0001]
    # architectures = [[X.shape[1], 15, y.shape[1]],
    #                  [X.shape[1], 35, y.shape[1]],
    #                  [X.shape[1], 35, 15, y.shape[1]],
    #                  [X.shape[1], 50, 35, y.shape[1]],
    #                  [X.shape[1], 100, 25, y.shape[1]]]

    # for i in range(len(architectures)):
    #     MultiErrorVsEpochTestPlotter(f"EX3_C_TRAINING_MULTI_LR_{i}.png",
    #                                  f"Number recognition: error evolution. Sigmoid. \nArch.: {architectures[i]}. "
    #                                  f"Test cases: {TEST_COUNT}",
    #                                  "Epoch",
    #                                  "Error (MSE)",
    #                                  "LR",
    #                                  learning_rates).plot(
    #         (lambda: [MultiLayerPerceptron(architectures[i],
    #                                        MAX_EPOCHS,
    #                                        FalseCutCondition(),
    #                                        SigmoidActivationFunction(),
    #                                        GradientDescentOptimization(lr)
    #                                        ).train_batch(X, y) for lr in learning_rates])
    #     )

    arch = [X.shape[1], 100, 25, y.shape[1]]
    # mlp = MultiLayerPerceptron(arch,
    #                            100000,
    #                            MSECutCondition(0.0001),
    #                            SigmoidActivationFunction(),
    #                            MomentumOptimization(alpha=0.9, learning_rate=0.01, architecture=arch)
    #                            )
    # mlp.train_batch(X, y)
    #
    # mismatches = [0 for _ in range(len(noises))]
    # for i in range(len(noises)):
    #     X_test, y_test = utils.noisy_set(X, noises[i], testing_size)
    #     for index, test in enumerate(mlp.predict(X_test)):
    #         if y_test[index] != np.argmax(test):
    #             mismatches[i] += 1
    #
    # mismatches = [(100 * m) / testing_size for m in mismatches]
    # noises = [str(round(n, 1)) for n in noises]
    #
    # fig, ax = plt.subplots()
    # ax.bar(noises, mismatches, align='center')
    #
    # # Add labels to the top of each bar
    # for i, v in enumerate(mismatches):
    #     ax.text(i, v + max(mismatches) * 0.01, f"{v:.1f}%", ha='center')
    #
    # ax.set_xticks(noises)
    # ax.set_xlabel('Noises')
    # ax.set_ylabel('Mismatch Count')
    # ax.set_title(f'Mismatch Counts for Different Noises. Arch.: [35, 100, 25, 10].\n'
    #              f'LR: 0.01. Momentum. Alpha: 0.9. Sigmoid. Test cases: 1000')
    # plt.savefig('figs/EX3C_Generalization.png')

    lrs = [0.3, 0.6, 0.9]

    functions = [TangentActivationFunction(beta=1), SigmoidActivationFunction(),
                 LogisticActivationFunction(beta=1)]
    af_labels = ["Tan(1)", "Sig", "Log(1)"]

    for i in range(len(lrs)):
        MultiErrorVsEpochTestPlotter(f"EX3C_AF_COMPARISON_{i}.png",
                                     f"Error vs. Epoch for different activation functions\n"
                                     f"Arch.: {arch}. LR: {lrs[i]}. Gradient. Test cases = {TEST_COUNT}",
                                     "Epoch",
                                     "Error (MSE)",
                                     "AF",
                                     af_labels).plot(
            lambda: [MultiLayerPerceptron(arch,
                                          MAX_EPOCHS,
                                          FalseCutCondition(),
                                          af,
                                          GradientDescentOptimization(lrs[i])
                                          ).train_batch(X, y) for af in functions]
        )


if __name__ == "__main__":
    plots_e3c()
