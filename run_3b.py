import numpy as np

import utils
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import MomentumOptimization
import matplotlib.pyplot as plt


def run_3_b():
    settings = utils.get_settings()
    cut_condition = utils.get_cut_condition(settings)
    activation_method = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    epochs = utils.get_epochs(settings)

    X = utils.get_numbers(settings)
    EXPECTED = np.array([[1 - 2 * (i % 2)] for i in range(len(X))])
    train_index = int(utils.get_train_ratio(settings) * (X.shape[0] - 1)) + 1

    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.3f}'.format})

    perceptron = MultiLayerPerceptron([X.shape[1], 5, EXPECTED.shape[1]],
                                      epochs,
                                      cut_condition,
                                      activation_method,
                                      MomentumOptimization(0.3,
                                                           utils.get_settings()["optimization_method"]["learning_rate"],
                                                           [X.shape[1], 5, EXPECTED.shape[1]]))
    print(f"training {train_index} numbers")

    print(f"Training finished in {len(perceptron.train_batch(X[:train_index, :], EXPECTED[:train_index, :]))} epochs.")

    ans = perceptron.predict(X)
    for index, test in enumerate(ans):
        result = "odd" if test < 0.5 else "even"
        print(f"Results for {index}: {test} -> {result}")
    print(["odd" if test < 0.5 else "even" for test in ans])

    result = [-1 if test < 0.5 else 1 for test in ans]

    return result == EXPECTED.flatten()


if __name__ == "__main__":
    counts = np.zeros(10)
    for i in range(100):
        results = run_3_b()
        counts[results] += 1

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    plt.bar(labels, counts)
    plt.xlabel('Numbers')
    plt.ylabel('Correct predictions')
    plt.title('Correct predictions of 100 runs with 80% train ratio')
    plt.ylim(0, 100)

    plt.show()
