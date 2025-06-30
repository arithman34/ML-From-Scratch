import numpy as np
from src.utils.backend import initialize_component


class Criteria:
    @staticmethod
    def __call__(y):
        raise NotImplementedError("The method '__call__' must be overridden in subclass.")


class Gini(Criteria):
    @staticmethod
    def __call__(y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)


class SquaredError(Criteria):
    @staticmethod
    def __call__(y):
        return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0


_CLF_CRITERIA = {
    "gini": Gini
}


_REG_CRITERIA = {
    "squared_error": SquaredError
}


def _initialize_criterion(criterion, is_classification=True):
    if is_classification:
        mapping = _CLF_CRITERIA
    else:
        mapping = _REG_CRITERIA

    return initialize_component("criterion", criterion, mapping, Criteria)