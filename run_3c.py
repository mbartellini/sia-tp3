import numpy as np

import utils
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import MomentumOptimization


def run_3_c():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    # optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = utils.get_numbers(settings)
    EXPECTED = np.array([[int(i == j) for j in range(10)] for i in range(len(X))])
    optimization_method = \
        MomentumOptimization(learning_rate=0.01, alpha=0.9, architecture=[X.shape[1], 100, 25, EXPECTED.shape[1]])

    perceptron = MultiLayerPerceptron([X.shape[1], 100, 25, EXPECTED.shape[1]],
                                      epochs,
                                      cut_condition,
                                      activation_method,
                                      optimization_method)
    print(f"Training finished in {len(perceptron.train_batch(X, EXPECTED))} epochs.")

    np.set_printoptions(suppress=True,
                        formatter={'float_kind': '{:0.3f}'.format})

    noise = utils.get_noise(settings)
    ts = utils.get_testing_size(settings)
    X_test, y_test = utils.noisy_set(X, noise, ts)

    for index, test in enumerate(perceptron.predict(X_test)):
        if y_test[index] != np.argmax(test):
            print(f"Mismatch for test {index}: {test} -> Predicted: {np.argmax(test)}, Expected: {y_test[index]}")


if __name__ == "__main__":
    run_3_c()
