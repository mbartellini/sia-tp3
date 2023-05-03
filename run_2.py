import utils

from src.simple_perceptron import SimplePerceptron
from src.activation_method import IdentityActivationFunction
from src.error import mse


def run_2():
    settings = utils.get_settings()
    X, y = utils.parse_csv(settings)
    activation_function = utils.get_activation_function(settings)
    optimization_method = utils.get_optimization_method(settings)
    cut_condition = utils.get_cut_condition(settings)
    epochs = utils.get_epochs(settings)

    # Aprendizaje
    p = [SimplePerceptron(X[0].size, epochs, cut_condition, IdentityActivationFunction(), optimization_method),
         SimplePerceptron(X[0].size, epochs, cut_condition, activation_function, optimization_method)]

    res = []
    for i in range(len(p)):
        res.append(p[i].train_batch(X, y))

    labels = ["linear", "nonlinear"]
    for i in range(len(p)):
        print(labels[i])
        print(mse(y - p[i].predict(X)))
        print("-" * 20)


if __name__ == "__main__":
    run_2()
