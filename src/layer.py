class Layer:
    def __init__(self, neurons):
        self._neurons = neurons

    @property
    def neurons(self):
        return self._neurons
