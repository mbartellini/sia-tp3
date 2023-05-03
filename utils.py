import json
import numpy as np
import sys

from tqdm import tqdm

from src.cut_condition import CutCondition, AccuracyCutCondition, AbsoluteValueCutCondition, MSECutCondition
from src.activation_method import ActivationMethod, StepActivationFunction, LogisticActivationFunction, \
    TangentActivationFunction, IdentityActivationFunction, SigmoidActivationFunction
from src.optimization_method import OptimizationMethod, MomentumOptimization, GradientDescentOptimization
from numpy import ndarray

import matplotlib.pyplot as plt
from PIL import Image

FRAME_DURATION = 100  # ms
LAST_FRAME_DURATION = 20  # Frames


def get_settings():
    if len(sys.argv) < 2:
        print("Config file argument not found")
        exit(1)

    path = sys.argv[1]
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
            return SigmoidActivationFunction()
        case "tangent":
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


def parse_csv(settings) -> tuple[ndarray, ndarray]:
    path = settings["path"]
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    inputs = np.array(data[:, :-1])  # All rows, all columns except the last (output)
    outputs = np.array(data[:, -1])  # All rows, last column
    return inputs, outputs


def scale(data: ndarray, limits: tuple[float, float]) -> ndarray:
    scaled = []
    x_min, x_max = min(data), max(data)
    for i in range(len(data)):
        scaled.append((((data[i] - x_min) * (limits[1] - limits[0])) / (x_max - x_min)) + limits[0])
    return np.array(scaled)


def get_train_ratio(settings) -> float:
    return settings["train_ratio"]


def animate(weights_history, data, expected, name, lr, frame_duration=FRAME_DURATION, last_frame_duration=LAST_FRAME_DURATION):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=expected)

    line, = ax.plot([], [], lw=2)
    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))

    ax.set_title(f"Learning Rate: {lr} | Epoch 0")

    # Initialize a list to store each frame of the animation
    frames = []

    for i in tqdm(range(len(weights_history))):
        line.set_xdata([-1.2, 1.2])
        line.set_ydata((-weights_history[i][0] - weights_history[i][1] * np.array([-1.2, 1.2])) / weights_history[i][2])

        ax.set_title(f"Learning Rate: {lr} | Epoch {i}")

        # Draw the current frame and add it to the list of frames
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        frames.append(frame)
        plt.close(fig)

    # Duplicate the last frame and append it to the end of the frames list
    last_frame = frames[-1]
    for _ in range(last_frame_duration):
        frames.append(last_frame)

    # Save the frames as a GIF file
    frames[0].save(name, format='GIF', save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)
