import unittest
import numpy as np
from machine_learning_algorithms.loss_functions.mean_squared_error import (
    mean_squared_error,
)


class TestMeanSquareError(unittest.TestCase):
    def test_same_predictions_and_truths(self):
        # Test case to check if mean_squared_error returns 0.0 when predictions and truths are the same
        predictions = np.array([1, 2, 3])
        truths = np.array([1, 2, 3])
        self.assertEqual(mean_squared_error(predictions, truths), 0.0)

    def test_different_predictions_and_truths(self):
        # Test case to check if mean_squared_error returns the correct value when predictions and truths are different
        predictions = np.array([1, 2, 3])
        truths = np.array([4, 5, 6])
        self.assertEqual(mean_squared_error(predictions, truths), 9.0)

    def test_empty_arrays(self):
        # Test case to check if mean_squared_error returns 0.0 when predictions and truths are empty arrays
        predictions = np.array([])
        truths = np.array([])
        self.assertEqual(mean_squared_error(predictions, truths), 0.0)

    def test_arrays_with_different_lengths(self):
        # Test case to check if mean_squared_error raises a ValueError when predictions and truths have different lengths
        predictions = np.array([1, 2, 3])
        truths = np.array([1, 2])
        with self.assertRaises(ValueError):
            mean_squared_error(predictions, truths)


if __name__ == "__main__":
    unittest.main()
