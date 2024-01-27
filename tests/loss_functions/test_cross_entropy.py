import unittest
import numpy as np
from machine_learning_algorithms.loss_functions.cross_entropy import cross_entropy


class TestCrossEntropy(unittest.TestCase):
    def test_same_predictions_and_truths(self):
        # Test case to check if cross_entropy returns 0.0 when predictions and truths are the same
        predictions = [np.array([1, 0, 0, 0])]
        truths = [np.array([1, 0, 0, 0])]
        self.assertEqual(cross_entropy(predictions, truths), 0.0)

    def test_different_predictions_and_truths(self):
        # Test case to check if cross_entropy returns the correct value when predictions and truths are different
        predictions = [np.array([0.8, 0.1, 0.05, 0.05])]
        truths = [np.array([0, 1, 0, 0])]
        self.assertAlmostEqual(
            cross_entropy(predictions, truths), -np.log(0.1), places=10
        )

    def test_empty_arrays(self):
        # Test case to check if cross_entropy returns 0.0 when predictions and truths are empty arrays
        predictions = []
        truths = []
        with self.assertRaises(ValueError):
            cross_entropy(predictions, truths)

    def test_arrays_with_different_lengths(self):
        # Test case to check if cross_entropy raises a ValueError when predictions and truths have different lengths
        predictions = [np.array([0.8, 0.1, 0.05, 0.05])]
        truths = [np.array([1, 0])]
        with self.assertRaises(ValueError):
            cross_entropy(predictions, truths)


if __name__ == "__main__":
    unittest.main()
