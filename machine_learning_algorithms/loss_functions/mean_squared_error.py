import numpy as np


def mean_squared_error(predictions: np.array, truths: np.array) -> float:
    """
    Calculate the mean square error between two arrays.

    Args:
        predictions (np.array): Array containing predicted values.
        truths (np.array): Array containing true values.

    Returns:
        float: Mean square error between the two arrays.
    """

    # Check if the arrays are empty
    if len(predictions) == 0 and len(truths) == 0:
        return 0.0

    # Check if the arrays have the same length
    if len(predictions) != len(truths):
        raise ValueError("The predictions and truths should have the same length")

    # Calculate the difference between predictions and truths
    difference = predictions - truths

    # Square the difference
    squared_difference = np.square(difference)

    # Calculate the mean of squared differences
    mean_squared_error = np.mean(squared_difference)

    return mean_squared_error
