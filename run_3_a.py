import sys

import numpy as np

import utils
from src.error_method import MeanSquaredErrorMethod
from src.multi_layer_perceptron import MultiLayerPerceptron

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


def run_3_a():
    if len(sys.argv) < 2:
        print("Config file argument not found")
        exit(1)

    config_path = sys.argv[1]
    settings = utils.get_settings(config_path)
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)
    error_method = MeanSquaredErrorMethod()  # Not used for the moment

    perceptron = MultiLayerPerceptron(activation_method, error_method, 0.1, epochs, [2, 3, 2, 1],
                                      optimization_method, cut_condition)
    print(f"Training finished in {perceptron.train_batch(X, y)} epochs.")

    for test in range(len(X)):
        print(f"{X[test][0]} & {X[test][1]} = {perceptron.test(X[test])}")


if __name__ == "__main__":
    run_3_a()