import numpy as np

from typing import List

from collections import Counter

# TODO: Add new distance metrics


class knn_classifier:
    def __init__(
        self, training_data: np.ndarray, classification_column: np.ndarray, k: int = 1
    ) -> None:
        """
        Initializes the KNN classifier.

        Args:
            training_data (np.ndarray): The training data.
            classification_column (np.ndarray): The classification column.
            k (int, optional): The number of nearest neighbors to consider. Defaults to 1.
        """
        self.training_data = training_data
        self.classification_column = classification_column
        self.k = k

    def find_k_nearest_neighbours(self, input_data_point: np.ndarray) -> List[int]:
        """
        Finds the k nearest neighbors for a given input data point.

        Args:
            input_data_point (np.ndarray): The input data point.

        Returns:
            List[int]: The classification labels of the k nearest neighbors.
        """
        distances = {}
        for index, data_point in enumerate(self.training_data):
            distances[index] = np.linalg.norm(data_point - input_data_point)

        # Sort the distances in ascending order
        sorted_distances: List[tuple] = sorted(distances.items(), key=lambda x: x[1])

        # Get the k nearest neighbours
        k_nearest_neighbours: List[int] = [
            self.classification_column[index, 0]
            for index in sorted_distances[: self.k][0]
        ]

        class_count = Counter(k_nearest_neighbours)
        most_common_class: List[tuple] = class_count.most_common(1)[0][0]

        return most_common_class

    def predict(self, input_data: np.ndarray) -> List[int]:
        """
        Predicts the classification labels for a given input data.

        Args:
            input_data (np.ndarray): The input data.

        Returns:
            List[int]: The predicted classification labels.
        """
        predictions = []
        for data_point in input_data:
            predictions.append(self.find_k_nearest_neighbours(data_point))
        return predictions
