import sys
import utils
import numpy as np

from src.simple_perceptron import SimplePerceptron
from src.activation_method import StepActivationFunction
from src.error import mse


def run_1():
    if len(sys.argv) < 2:
        print("Config file argument not found")
        exit(1)

    config_path = sys.argv[1]
    settings = utils.get_settings(config_path)
    cut_condition = utils.get_cut_condition(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    expected = np.array([[-1, -1, -1, 1], [-1, 1, 1, -1]])

    perceptrons = []
    for i in range(len(expected)):
        perceptrons.append(SimplePerceptron(2, epochs, cut_condition, StepActivationFunction(), optimization_method))

    res = []
    for i in range(len(expected)):
        res.append(perceptrons[i].train_batch(X, expected[i]))


if __name__ == "__main__":
    run_1()
