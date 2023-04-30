from abc import ABC

import numpy as np


class UpdateMethod(ABC):
    def process_prediction(self, weights, dw):
        raise NotImplementedError()

    def process_epoch(self, weights):
        raise NotImplementedError()


class OnlineUpdateMethod(UpdateMethod):

    def process_prediction(self, weights, dw):
        return np.add(weights, dw)

    def process_epoch(self, weights):
        pass


class BatchUpdateMethod(UpdateMethod):
    def __init__(self):
        self._cum_dw = None

    def process_prediction(self, weights, dw):
        if self._cum_dw is None:
            self._cum_dw = np.zeros(len(dw))
        self._cum_dw = np.add(self._cum_dw, dw)
        return weights

    def process_epoch(self, weights):
        ans = np.add(weights, self._cum_dw)
        self._cum_dw = np.zeros(self._cum_dw)
        return ans
