from abc import ABC


class ActivationMethod(ABC):
    @staticmethod
    def evaluate(x: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def d_evaluate(x: float) -> float:
        raise NotImplementedError()


class StepActivationFunction(ActivationMethod):
    @staticmethod
    def evaluate(x: float) -> float:
        return 1 if x >= 0 else -1

    @staticmethod
    def d_evaluate(x: float) -> float:
        return 1


class IdentityActivationFunction(ActivationMethod):
    @staticmethod
    def evaluate(x: float) -> float:
        return x

    @staticmethod
    def d_evaluate(x: float) -> float:
        return 1
