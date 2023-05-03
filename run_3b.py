import sys

import numpy as np
from numpy import ndarray

import utils
from src.multi_layer_perceptron import MultiLayerPerceptron


def get_numbers(path: str) -> ndarray[float]:
    with open(path, "r") as f:
        list_of_lines = f.read().splitlines()
        list_of_lines_without_spaces = [np.fromstring(line, dtype=int, sep=' ') for line in list_of_lines]
        numbers = np.array(
            list(zip(*(iter(list_of_lines_without_spaces),) * (len(list_of_lines_without_spaces) // 10))))
    return np.array([number.ravel() for number in numbers])


def run_3_a():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = get_numbers("data/TP3-ej3-digitos.txt")
    EXPECTED = np.array([[1 - 2 * (i % 2)] for i in range(len(X))])
    print(X)

    perceptron = MultiLayerPerceptron([X.shape[1], 100, 25, EXPECTED.shape[1]],
                                      epochs,
                                      cut_condition,
                                      activation_method,
                                      optimization_method)
    print(f"Training finished in {perceptron.train_batch(X, EXPECTED)} epochs.")

    ans = perceptron.predict(X)
    print(ans)


if __name__ == "__main__":
    run_3_a()
