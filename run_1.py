import sys
import utils
import numpy as np

from plot import plots_e1
from src.simple_perceptron import SimplePerceptron
from src.activation_method import StepActivationFunction
from src.error import mse


def run_1():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    expected = np.array([[-1, -1, -1, 1], [-1, 1, 1, -1]])

    perceptrons = []
    for i in range(len(expected)):
        perceptrons.append(SimplePerceptron(2, epochs, cut_condition, StepActivationFunction(), optimization_method))

    (error_history_and, weight_history_and) = perceptrons[0].train_batch(X, expected[0])
    (error_history_xor, weight_history_xor) = perceptrons[1].train_batch(X, expected[1])

    print(f"Training finished for logical 'and' in {len(error_history_and)} epochs.")
    print(f"Training finished for logical 'xor' in {len(error_history_xor)} epochs.")

    np.set_printoptions(suppress=True,
                        formatter={'float_kind': '{:0.3f}'.format})

    for index, test in enumerate(perceptrons[0].predict(X)):
        print(f"Results for test 'and' {index}: {test} -> Expected: {expected[0][index]} "
              f"-> Error: {mse(np.array([test - expected[0][index]]))}")

    for index, test in enumerate(perceptrons[1].predict(X)):
        print(f"Results for test 'xor' {index}: {test} -> Expected: {expected[1][index]} "
              f"-> Error: {mse(np.array([test - expected[1][index]]))}")

    print("Plotting...")
    utils.animate(weight_history_and, X, expected[0], "and_animation.gif", optimization_method.learning_rate, frame_duration=200, last_frame_duration=2)
    # utils.animate(weight_history_xor, X, expected[1], "xor_animation.gif", optimization_method.learning_rate, frame_duration=10, last_frame_duration=1)
    plots_e1()


if __name__ == "__main__":
    run_1()
