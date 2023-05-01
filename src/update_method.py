from abc import ABC

import numpy as np
from numpy import ndarray


class UpdateMethod(ABC):
    def process_prediction(self, weights: ndarray[float], dw: ndarray[float]):
        raise NotImplementedError()

    def process_epoch(self, weights: ndarray[float]):
        raise NotImplementedError()


class OnlineUpdateMethod(UpdateMethod):

    def process_prediction(self, weights: ndarray[float], dw: ndarray[float]):
        return np.add(weights, dw)

    def process_epoch(self, weights: ndarray[float]):
        pass


class BatchUpdateMethod(UpdateMethod):
    def __init__(self):
        self._cum_dw = None

    def process_prediction(self, weights: ndarray[float], dw: ndarray[float]):
        if self._cum_dw is None:
            self._cum_dw = np.zeros(len(dw))
        self._cum_dw = np.add(self._cum_dw, dw)
        return weights

    def process_epoch(self, weights: ndarray[float]):
        ans = np.add(weights, self._cum_dw)
        self._cum_dw = None
        return ans
