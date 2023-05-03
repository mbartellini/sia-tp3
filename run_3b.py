import sys

import numpy as np

import utils
from src.multi_layer_perceptron import MultiLayerPerceptron


def run_3_a():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = utils.get_numbers(settings)
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
