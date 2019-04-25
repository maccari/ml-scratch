import numpy as np


def accuracy_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """ """
    return (predictions == targets).sum() / len(targets)
