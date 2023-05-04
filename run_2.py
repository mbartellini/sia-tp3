import utils

from src.simple_perceptron import SimplePerceptron
from src.activation_method import IdentityActivationFunction
from src.error import mse
import copy
import numpy as np


def learning_test(X, y, p, limits):
    y = [y, utils.scale(y, limits)]
    res = [p[0].train_batch(X, y[0]), p[1].train_batch(X, y[1])]

    print("linear")
    print(mse(y[0] - p[0].predict(X)))
    print("non-linear")
    print(mse(y[1] - p[1].predict(X)))


def generalization_test(X, y, p, limits, train_ratio):
    data = np.column_stack((X, y))
    np.random.shuffle(data)

    train_ratio = 0.8
    split_index = int(len(data) * train_ratio)

    X_train, y_train = data[:split_index, :-1], data[:split_index, -1]
    X_test, y_test = data[split_index:, :-1], data[split_index:, -1]
    y_train = [y_train, utils.scale(y_train, limits)]
    y_test = [y_test, utils.scale(y_test, limits)]

    res = [p[0].train_batch(X_train, y_train[0]), p[1].train_batch(X_train, y_train[1])]

    print("linear")
    print(mse(y_test[0] - p[0].predict(X_test)))
    print("non-linear")
    print(mse(y_test[1] - p[1].predict(X_test)))


def run_2():
    settings = utils.get_settings()
    X, y = utils.parse_csv(settings)
    activation_function = utils.get_activation_function(settings)
    limits = activation_function.limits()
    optimization_method = utils.get_optimization_method(settings)
    cut_condition = utils.get_cut_condition(settings)
    epochs = utils.get_epochs(settings)

    # Aprendizaje
    p1 = [SimplePerceptron(X[0].size, epochs, cut_condition, IdentityActivationFunction(), optimization_method),
          SimplePerceptron(X[0].size, epochs, cut_condition, activation_function, optimization_method)]
    p2 = copy.deepcopy(p1)

    print("Leaning with whole test: ")
    learning_test(X, y, p1, limits)
    print("-" * 20)
    print("Training testing: 80-20")
    generalization_test(X, y, p2, limits, utils.get_train_ratio(settings))


if __name__ == "__main__":
    run_2()
