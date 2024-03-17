from typing import List

from machine_learning_algorithms.neural_network_algorithms.perceptron.layer import (
    Perceptron_layer,
)


class Perceptron_network:
    def __init__(self, depths: List[int]) -> None:
        self.layers = [Perceptron_layer(depth) for depth in depths]
