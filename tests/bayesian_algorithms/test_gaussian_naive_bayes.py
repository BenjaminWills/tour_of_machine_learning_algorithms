import unittest
import numpy as np
from machine_learning_algorithms.bayesian_algorithms.gaussian_naive_bayes.gaussian_naive_bayes import (
    gaussian_naive_bayes,
    calculate_gaussian_mean_and_std,
)


class TestGaussianNaiveBayes(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        self.independent_variables = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.classification_column = np.array([0, 1, 0])
        self.gnb = gaussian_naive_bayes(
            self.independent_variables, self.classification_column
        )

    def test_calculate_gaussian_mean_and_std(self):
        # Test case to check if calculate_gaussian_mean_and_std returns the correct mean and std for each class
        expected_mean_and_std = {
            0: {"mean": 4.0, "std": 3.0},
            1: {"mean": 4.0, "std": 0.0},
        }

        self.assertEqual(
            calculate_gaussian_mean_and_std(
                self.independent_variables[:, 0], self.classification_column
            ),
            expected_mean_and_std,
        )

    def test_calculate_posteriors(self):
        # Test case to check if calculate_posteriors returns the correct mean and std for each independent variable and class
        expected_posteriors = {
            0: {0: {"mean": 4.0, "std": 3.0}, 1: {"mean": 4.0, "std": 0.0}},
            1: {0: {"mean": 5.0, "std": 3.0}, 1: {"mean": 5.0, "std": 0.0}},
            2: {0: {"mean": 6.0, "std": 3.0}, 1: {"mean": 6.0, "std": 0.0}},
        }
        self.assertEqual(self.gnb.posteriors, expected_posteriors)

    def test_calculate_class_probability(self):
        # Test case to check if calculate_class_probability returns the correct probability for a given class and input data
        input_data = np.array([2, 1, 3])
        class_ = 0
        expected_probability = 0.00031302462861848297
        self.assertEqual(
            self.gnb.calculate_class_probability(input_data, class_),
            expected_probability,
        )


if __name__ == "__main__":
    unittest.main()
