import numpy as np

from typing import List, Callable


class network:
    def __init__(
        self, activation_functions: List[Callable], num_perceptrons: List[int]
    ) -> None:
        """
        Initializes a neural network.

        Args:
            activation_functions (List[Callable]): List of activation functions for each layer.
            num_perceptrons (List[int]): List of the number of perceptrons in each layer.
        """
        self.activation_functions = activation_functions
        self.num_perceptrons = num_perceptrons

        # Initialise weights and biases
        self.initialise_weights()

    def feedforward(self, inputs: np.array) -> np.array:
        """
        Performs a feedforward pass through the network.

        Args:
            inputs (np.array): Input data.

        Returns:
            np.array: Output of the network.
        """
        # Forward propagate the inputs through the network
        for i in range(len(self.num_perceptrons) - 1):
            activation_function = self.activation_functions[i]

            linear_product = np.dot(self.weights[i], inputs) + self.biases[i + 1]
            inputs = activation_function(linear_product)
        return inputs

    def initialise_weights(self) -> None:
        """
        Initializes the weights and biases of the network.
        """
        # A column in added to the weights for the bias

        # For each perception in the layer we need to generate a weight matrix that has the dimension
        # of the number of neruons in the next layer by the number of neurons in the current layer
        self.weights = [
            np.random.rand(self.num_perceptrons[i + 1], self.num_perceptrons[i])
            for i in range(len(self.num_perceptrons) - 1)
        ]
        self.biases = [
            np.random.rand(self.num_perceptrons[i])
            for i in range(len(self.num_perceptrons))
        ]
