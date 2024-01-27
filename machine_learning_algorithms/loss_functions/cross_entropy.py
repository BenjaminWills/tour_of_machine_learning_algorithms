import numpy as np


def cross_entropy(predictions: np.array, truths: np.array):
    # Calculate the number of datapoints
    num_data_points = len(predictions)

    # Calculate the loss
    loss = 0
    for prediction, truth in zip(predictions, truths):
        # Note that this is a dot as sometimes predicitons
        # and truths can be matrices or arrays
        # e.g
        # truth     : [1, 0, 0, 0]
        # prediction: [0.8, 0.1, 0.05, 0.05]

        # Take the log of predictions
        log_predictions = np.log(prediction)
        loss -= np.dot(truth, log_predictions)

    # Calculate average log loss.
    average_loss = loss / num_data_points

    return average_loss
