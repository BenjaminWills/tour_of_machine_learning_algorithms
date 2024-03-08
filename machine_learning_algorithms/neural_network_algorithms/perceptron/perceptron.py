import numpy as np

from typing import Callable


def step_function(x: float) -> int:
    return 1 if x >= 0 else 0


class Perceptron:
    def __init__(
        self, weights: np.ndarray, bias: float, activation_function: Callable
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def predict(self, inputs: np.ndarray) -> int:
        return step_function(
            self.activation_function(np.dot(inputs, self.weights) + self.bias)
        )
