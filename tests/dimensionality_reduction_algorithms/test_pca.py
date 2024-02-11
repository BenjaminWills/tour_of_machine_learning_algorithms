import unittest
import numpy as np
from machine_learning_algorithms.dimensionality_reduction_algorithms.principal_component_anlaysis.pca import (
    pca,
)

from sklearn.datasets import load_iris

data = load_iris()
independent_variables = data.data


class TestPCA(unittest.TestCase):
    def setUp(self):
        # Create an instance of the logistic regressor
        self.data = independent_variables
        self.pca = pca(self.data)

    def test_valid_data(self):
        # Test case to check if pca returns the correct projected data for valid input
        num_components = 2
        reduced_data = self.pca.reduce_dimensionality(num_components)
        self.assertEqual(reduced_data.shape, (data.shape[0], num_components))

    def test_num_components_greater_than_features(self):
        # Test case to check if pca raises a ValueError when the number of components is greater than the number of features
        num_components = 5
        with self.assertRaises(ValueError):
            self.pca.reduce_dimensionality(num_components)

    def test_num_components_greater_than_samples(self):
        # Test case to check if pca raises a ValueError when the number of components is greater than the number of samples
        num_components = 151
        with self.assertRaises(ValueError):
            self.pca.reduce_dimensionality(num_components)

    def test_invalid_data_type(self):
        # Test case to check if pca raises a TypeError when the data is not a numpy array
        data = "a"
        num_components = 2
        with self.assertRaises(TypeError):
            pca(data).reduce_dimensionality(num_components)

    def test_invalid_num_components_type(self):
        # Test case to check if pca raises a TypeError when the number of components is not an integer
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        num_components = 2.5
        with self.assertRaises(TypeError):
            pca(data).reduce_dimensionality(num_components)


if __name__ == "__main__":
    unittest.main()
