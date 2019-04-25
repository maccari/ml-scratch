import numpy as np


def step(outputs: np.ndarray) -> np.ndarray:
    """ """
    return (outputs > 0).astype(int)


def sigmoid(output: float, derivative: bool = False) -> float:
    """ """
    out = 1 / (1 + np.e ** -output)
    if derivative:
        out = out * (1 - out)
    return out
