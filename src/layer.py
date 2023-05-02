from numpy import ndarray


class Layer:
    def __init__(self, neurons: ndarray[float]):
        self._neurons = neurons

    @property
    def neurons(self) -> ndarray[float]:
        return self._neurons
