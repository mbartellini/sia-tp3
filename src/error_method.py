from abc import ABC
from numbers import Number

from numpy import ndarray


class ErrorMethod(ABC):
    @staticmethod
    def error(expected_output, real_output) -> Number:
        raise NotImplementedError()

    @staticmethod
    def d_error(expected_output, real_output, d_real_output) -> Number:
        raise NotImplementedError()


class MeanSquaredErrorMethod(ErrorMethod):
    @staticmethod
    def error(expected_output: ndarray[float], real_output: ndarray[float]) -> Number:
        g_error = 0
        for i in range(expected_output):
            g_error += (expected_output[i] - real_output[i]) ** 2
        g_error /= 2
        return g_error

    @staticmethod
    def d_error(expected_output: ndarray[float], real_output: ndarray[float], d_real_output: ndarray[float]) -> Number:
        return (expected_output - real_output) * d_real_output
