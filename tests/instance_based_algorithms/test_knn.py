import unittest
import numpy as np
from machine_learning_algorithms.instance_based_algorithms.k_means.knn import (
    knn_classifier,
)
from sklearn.datasets import load_iris

regression = load_iris()
independent_variable = regression.data
dependent_variable = regression.target


class TestKNNClassifier(unittest.TestCase):
    def setUp(self):
        # Create a sample training dataset
        self.k = 3
        self.knn = knn_classifier(independent_variable, dependent_variable)

    def test_find_k_nearest_neighbours(self):
        # Test case to check if the k nearest neighbors are correctly identified
        input_data_point = np.array([5.1, 3.5, 1.4, 0.2])
        most_common_neighbour_class = self.knn.find_k_nearest_neighbours(
            input_data_point, self.k
        )
        self.assertEqual(0, most_common_neighbour_class)

    def test_predict(self):
        # Test case to check if the classification labels are correctly predicted
        input_data = np.array(
            [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8], [7.0, 3.2, 4.7, 1.4]]
        )
        expected_predictions = [
            0,
            2,
            1,
        ]  # The first and third data points belong to class 0, the second data point belongs to class 1
        predictions = self.knn.predict(input_data, k=3)
        self.assertEqual(predictions, expected_predictions)


if __name__ == "__main__":
    unittest.main()
