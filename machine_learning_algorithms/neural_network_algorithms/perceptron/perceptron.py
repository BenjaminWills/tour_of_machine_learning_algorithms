import numpy as np

from typing import Callable


class Perceptron:
    def __init__(self, weights: np.ndarray, activation_function: Callable) -> None:
        """
        Initialize the Perceptron model.

        Args:
            weights (np.ndarray): The weight vector for the perceptron. The last element of the vector is the bias.
            activation_function (Callable): The activation function to be used.

        Returns:
            None
        """
        self.weights = weights[:-1]
        self.bias = weights[-1]
        self.activation_function = activation_function

    def predict(self, inputs: np.ndarray) -> int:
        """
        Predict the output for the given inputs.

        Args:
            inputs (np.ndarray): The input vector.

        Returns:
            int: The predicted output.
        """
        return self.activation_function(np.dot(inputs, self.weights) + self.bias)
