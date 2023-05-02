import json

from src.cut_condition import CutCondition, AccuracyCutCondition, AbsoluteValueCutCondition, MSECutCondition
from src.activation_method import ActivationMethod, StepActivationFunction, LogisticActivationFunction,  \
    TangentActivationFunction, IdentityActivationFunction
from src.optimization_method import OptimizationMethod, MomentumOptimization, GradientDescentOptimization


def get_settings(path: str):
    with open(path, "r") as f:
        settings = json.load(f)
    if settings is None:
        raise ValueError("Unable to open settings")
    return settings


def get_activation_function(settings) -> ActivationMethod:
    function = settings["activation_function"]["function"]
    match function:
        case "step":
            return StepActivationFunction()
        case "identity":
            return IdentityActivationFunction()
        case "logistic":
            beta = settings["activation_function"]["beta"]
            return LogisticActivationFunction(beta)
        case "sigmoid":
            beta = settings["activation_function"]["beta"]
            return TangentActivationFunction(beta)
        case _:
            raise ValueError("Unsupported activation function: " + function)


def get_cut_condition(settings) -> CutCondition:
    condition = settings["cut_condition"]["condition"]
    match condition:
        case "accuracy":
            return AccuracyCutCondition()
        case "absolute":
            return AbsoluteValueCutCondition()
        case "mse":
            eps = settings["cut_condition"]["eps"]
            return MSECutCondition(eps=eps)


def get_optimization_method(settings) -> OptimizationMethod:
    method = settings["optimization_method"]["method"]
    match method:
        case "gradient":
            lr = settings["optimization_method"]["learning_rate"]
            return GradientDescentOptimization(learning_rate=lr)
        case "momentum":
            lr = settings["optimization_method"]["learning_rate"]
            alpha = settings["optimization_method"]["alpha"]
            return MomentumOptimization(learning_rate=lr, alpha=alpha)
        case _:
            raise ValueError("Optimization method not supported: " + method)


def get_epochs(settings) -> int:
    return settings["epochs"]
