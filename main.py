import numpy as np

from src.activation_method import StepActivationFunction
from src.cut_condition import AccuracyCutCondition
from src.optimization_method import GradientDescentOptimization
from src.simple_perceptron import SimplePerceptron
from src.update_method import OnlineUpdateMethod


def main():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    update_method = OnlineUpdateMethod()
    cut_condition = AccuracyCutCondition(y.shape[0])
    activation_method = StepActivationFunction()
    optimization_method = GradientDescentOptimization()
    perceptron = SimplePerceptron(2, 1000, update_method, cut_condition, activation_method, optimization_method)
    periods = perceptron.train(X, y)

    print(periods)

    X_test = X
    y_pred = perceptron.predict(X_test)
    print(y_pred)


if __name__ == '__main__':
    main()
