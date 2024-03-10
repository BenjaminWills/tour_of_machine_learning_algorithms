import numpy as np

from typing import Callable


class Perceptron:
    def __init__(self, weights: np.ndarray, activation_function: Callable) -> None:
        self.weights = weights[:-1]
        self.bias = weights[-1]
        self.activation_function = activation_function

    def predict(self, inputs: np.ndarray) -> int:
        return self.activation_function(np.dot(inputs, self.weights) + self.bias)
