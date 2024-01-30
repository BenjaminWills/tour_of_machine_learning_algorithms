import unittest
import numpy as np
from machine_learning_algorithms.instance_based_algorithms.k_means.knn import (
    knn_classifier,
)


class TestKNNClassifier(unittest.TestCase):
    def setUp(self):
        # Create a sample training dataset
        self.training_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.classification_column = np.array([[0], [0], [1], [1], [1]])
        self.k = 3
        self.knn = knn_classifier(self.training_data, self.classification_column)

    def test_find_k_nearest_neighbours(self):
        # Test case to check if the k nearest neighbors are correctly identified
        input_data_point = np.array([3, 3])
        expected_neighbours = [0, 1, 0]  # The first three data points belong to class 0
        most_common_neighbour_class = self.knn.find_k_nearest_neighbours(
            input_data_point, self.k
        )
        self.assertEqual(0, most_common_neighbour_class)

    def test_predict(self):
        # Test case to check if the classification labels are correctly predicted
        input_data = np.array([[1, 2], [4, 5], [2, 3]])
        expected_predictions = [
            0,
            1,
            0,
        ]  # The first and third data points belong to class 0, the second data point belongs to class 1
        predictions = self.knn.predict(input_data)
        self.assertEqual(predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()
