import sys

import numpy as np

import utils
from run_3b import get_numbers
from src.multi_layer_perceptron import MultiLayerPerceptron


def run_3_a():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = get_numbers("data/TP3-ej3-digitos.txt")
    EXPECTED = np.array([[int(i == j) for j in range(10)] for i in range(len(X))])

    perceptron = MultiLayerPerceptron([X.shape[1], 100, 25, EXPECTED.shape[1]],
                                      epochs,
                                      cut_condition,
                                      activation_method,
                                      optimization_method)
    print(f"Training finished in {perceptron.train_batch(X, EXPECTED)} epochs.")

    np.set_printoptions(suppress=True,
                        formatter={'float_kind': '{:0.3f}'.format})
    for index, test in enumerate(perceptron.predict(X)):
        print(f"Results for test {index}: {test} -> {np.argmax(test)}")
    print([np.argmax(test) for test in perceptron.predict(X)])


if __name__ == "__main__":
    run_3_a()
