import numpy as np

import utils
from src.multi_layer_perceptron import MultiLayerPerceptron


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
                                      MomentumOptimization(utils.get_settings()["optimization_method"]["alpha"], utils.get_settings()["optimization_method"]["learning_rate"], [X.shape[1], 5, EXPECTED.shape[1]]))
    print(f"training {train_index} numbers")

    print(f"Training finished in {len(perceptron.train_batch(X[:train_index, :], EXPECTED[:train_index, :]))} epochs.")

    ans = perceptron.predict(X)
    for index, test in enumerate(ans):
        result = "odd" if test < 0.5 else "even"
        print(f"Results for {index}: {test} -> {result}")
    print(["odd" if test < 0.5 else "even" for test in ans])


if __name__ == "__main__":
    run_3_b()
