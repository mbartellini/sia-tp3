import numpy as np

from src.simple_perceptron import SimplePerceptron


def main():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    perceptron = SimplePerceptron(2)
    perceptron.train(X, y)

    X_test = X
    y_pred = perceptron.predict(X_test)
    print(y_pred)


if __name__ == '__main__':
    main()
