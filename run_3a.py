import numpy as np

import utils
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
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    perceptron = MultiLayerPerceptron([2, 2, 1],
                                      epochs,
                                      cut_condition,
                                      activation_method,
                                      optimization_method)
    print(f"Training finished in {len(perceptron.train_batch(X, EXPECTED))} epochs.")

    ans = perceptron.predict(X)
    for test in range(X.shape[0]):
        print(f"{X[test][0]} & {X[test][1]} = {ans[test]}")


if __name__ == "__main__":
    run_3_a()
