import numpy as np


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid of a float.

    Args:
        x (float): float to calculate the sigmoid of.

    Returns:
        float: Sigmoid of the float.
    """
    return 1 / (1 + np.exp(-x))
