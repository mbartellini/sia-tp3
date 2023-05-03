import numpy as np
import tensorflow as tf

from src.activation_method import LogisticActivationFunction
from src.cut_condition import AccuracyCutCondition
from src.multi_layer_perceptron import MultiLayerPerceptron
from src.optimization_method import MomentumOptimization


def main():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the training and testing data
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    # Convert the labels to one-hot encoding
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    print("Loaded MNIST dataset.")

    cut_condition = AccuracyCutCondition()
    activation_method = LogisticActivationFunction(0.5)
    optimization_method = MomentumOptimization()  # Not used for the moment
    perceptron = MultiLayerPerceptron([784, 12, 12, 10], 2000, cut_condition, activation_method, optimization_method)

    print("Training progress:")
    epochs = perceptron.train_batch(x_train[:80], y_train[:80])
    print("Training finished.")

    y_pred = perceptron.predict(x_test[:10])
    print(y_pred)
    print(y_test[:10])

    # X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    # y = np.array([[1], [1], [-1], [-1]])
    #
    # cut_condition = AccuracyCutCondition()
    # activation_method = TangentActivationFunction(0.5)
    # optimization_method = MomentumOptimization()  # Not used for the moment
    # error_method = MeanSquaredErrorMethod()  # Not used for the moment
    # lr = 1
    # perceptron = MultiLayerPerceptron(activation_method, error_method, lr, 1000, [2, 2, 1], optimization_method, cut_condition)
    # epochs = perceptron.train_batch(X, y)
    #
    # print(epochs)
    #
    # X_test = X
    # y_pred = perceptron.test(X_test)
    # print(y_pred)


if __name__ == '__main__':
    main()
