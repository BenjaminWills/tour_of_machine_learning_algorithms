import numpy as np

from machine_learning_algorithms.neural_network_algorithms.perceptron.perceptron import (
    Perceptron,
)


class Perceptron_layer:
    def __init__(self, depth: int, weights: np.array) -> None:
        self.depth = depth
        self.neurons = [Perceptron(weights[i]) for i in range(depth)]

    def set_weights(self, weights: np.array) -> None:
        # Set the weights for each neuron in the layer
        return Perceptron_layer(self.depth, weights)

    def predict(self, inputs: np.array) -> np.array:
        # Forward propagate the inputs through the layer
        return np.array([neuron.predict(inputs) for neuron in self.neurons])
