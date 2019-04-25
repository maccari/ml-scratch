import numpy as np


def cross_entropy(outputs: np.ndarray, targets: np.ndarray) -> float:
    """ Each target must be 0 or 1 """
    loss = sum(np.log(outputs) * targets)
    loss += sum(np.log(1 - outputs) * (1 - targets))
    return -loss / len(targets)
