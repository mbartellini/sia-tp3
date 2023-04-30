import numpy as np

from src.cut_condition import AccuracyCutCondition
from src.simple_perceptron import SimplePerceptron
from src.update_method import OnlineUpdateMethod


def main():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    update_method = OnlineUpdateMethod()
    cut_condition = AccuracyCutCondition(y.shape[0])
    perceptron = SimplePerceptron(2, update_method, cut_condition)
    periods = perceptron.train(X, y)

    print(periods)

    X_test = X
    y_pred = perceptron.predict(X_test)
    print(y_pred)


if __name__ == '__main__':
    main()
