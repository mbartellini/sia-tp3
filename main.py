import numpy as np

from src.activation_method import StepActivationFunction
from src.cut_condition import AccuracyCutCondition
from src.optimization_method import MomentumOptimization
from src.simple_perceptron import SimplePerceptron


def main():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    cut_condition = AccuracyCutCondition()
    activation_method = StepActivationFunction()
    optimization_method = MomentumOptimization()
    perceptron = SimplePerceptron(2, 1000, cut_condition, activation_method, optimization_method)
    epochs = perceptron.train_batch(X, y)

    print(epochs)

    X_test = X
    y_pred = perceptron.predict(X_test)
    print(y_pred)


if __name__ == '__main__':
    main()
