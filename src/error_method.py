from abc import ABC
from numbers import Number


class ErrorMethod(ABC):
    @staticmethod
    def error(expected_output, real_output) -> Number:
        raise NotImplementedError()

    @staticmethod
    def d_error(expected_output, real_output, d_real_output) -> Number:
        raise NotImplementedError()
