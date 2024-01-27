import numpy as np

from typing import List


"""
Cross entropy is a loss function commonly used in machine learning algorithms 
to measure the dissimilarity between predicted values and true values. 
It calculates the average log loss by taking the dot product of the true 
values and the logarithm of the predicted values. The logarithm is used 
to penalize larger differences between the predicted and true values. 
The cross entropy is then divided by the number of data points to obtain the average loss.
"""


def cross_entropy(predictions: List[np.array], truths: List[np.array]):
    """
    Calculate the cross entropy between two arrays.

    Args:
        predictions (List[np.array]): Array containing predicted values.
        truths (List[np.array]): Array containing true values.

    Returns:
        float: Cross entropy between the two arrays.
    """
    # Calculate the number of datapoints
    num_data_points = len(predictions)

    # If the lengths of the two arrays are not equal, raise a ValueError
    if num_data_points != len(truths):
        raise ValueError("The lengths of the two arrays should be equal")

    # If some of the arrays are empty, raise value error
    if num_data_points == 0:
        raise ValueError("Empty arrays are not allowed")

    # Calculate the loss
    loss = 0
    for prediction, truth in zip(predictions, truths):
        # Note that this is a dot as sometimes predicitons
        # and truths can be matrices or arrays
        # e.g
        # truth     : [1, 0, 0, 0]
        # prediction: [0.8, 0.1, 0.05, 0.05]

        # In the case of empty arrays, a value error is raised
        if len(prediction) == 0 or len(truth) == 0:
            raise ValueError("Empty arrays are not allowed")

        # Take the log of predictions
        log_predictions = np.log(prediction)

        # if nan in log_predictions, replace it with an arbitrarily large value
        # to avoid nan in the loss
        log_predictions = np.nan_to_num(log_predictions, nan=1000)

        loss -= np.dot(truth, log_predictions)

    # Calculate average log loss.
    average_loss = loss / num_data_points

    return average_loss
