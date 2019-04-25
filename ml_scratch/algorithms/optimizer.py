import numpy as np


def compute_gradient(
    inputs: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    loss_fn: str,
) -> np.ndarray:
    """ """
    if loss_fn == "cross_entropy":
        gradient = np.dot((predictions - targets), inputs) / inputs.shape[0]
    else:
        raise ValueError(f"Unknown loss_fn {loss_fn}")
    return gradient
