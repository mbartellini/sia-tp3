from abc import ABC


class OptimizationMethod(ABC):
    @staticmethod
    def adjust():
        raise NotImplementedError()

