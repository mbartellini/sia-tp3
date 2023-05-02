import numpy as np

from src.activation_method import TangentActivationFunction
from src.cut_condition import AccuracyCutCondition
from src.error_method import MeanSquaredErrorMethod
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import MomentumOptimization


def main():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([[-1], [-1], [-1], [1]])

    cut_condition = AccuracyCutCondition()
    activation_method = TangentActivationFunction(0.5)
    optimization_method = MomentumOptimization()  # Not used for the moment
    error_method = MeanSquaredErrorMethod()  # Not used for the moment
    lr = 100
    perceptron = MultiLayerPerceptron(activation_method, error_method, lr, 1000, [2, 3, 2, 1], optimization_method, cut_condition)
    epochs = perceptron.train_batch(X, y)

    print(epochs)

    X_test = X
    y_pred = perceptron.test(X_test)
    print(y_pred)


if __name__ == '__main__':
    main()
