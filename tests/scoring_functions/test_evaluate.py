import unittest
import numpy as np
from machine_learning_algorithms.scoring_functions.evaluate import evaluate


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.y_true = np.array([0, 1, 1, 2, 1, 0])
        self.y_pred = np.array([0, 1, 2, 0, 1, 1])
        self.evaluator = evaluate(self.y_true, self.y_pred)

    def test_confusion_matrix(self):
        # Test case to check if the confusion matrix is calculated correctly
        expected_confusion_matrix = np.array(
            [[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 0.0]]
        )
        np.testing.assert_array_equal(
            self.evaluator.confusion_matrix(), expected_confusion_matrix
        )

    def test_accuracy(self):
        # Test case to check if the accuracy is calculated correctly
        expected_accuracy = 0.5
        self.assertAlmostEqual(self.evaluator.accuracy(), expected_accuracy)

    def test_tp_fp_fn(self):
        # Test case to check if the true positives, false positives, and false negatives are calculated correctly
        expected_tp_fp_fn = {
            0: {"true_positive": 1.0, "false_positive": 1.0, "false_negative": 1.0},
            1: {"true_positive": 2.0, "false_positive": 1.0, "false_negative": 1.0},
            2: {"true_positive": 0.0, "false_positive": 1.0, "false_negative": 1.0},
        }
        self.assertEqual(self.evaluator.tp_fp_fn, expected_tp_fp_fn)

    def test_precision_and_recall(self):
        # Test case to check if the precision and recall are calculated correctly
        expected_precision_recall = {
            0: {"precision": 0.5, "recall": 0.5},
            1: {"precision": 0.6666666666666666, "recall": 0.6666666666666666},
            2: {"precision": 0.0, "recall": 0.0},
        }
        self.assertEqual(self.evaluator.precision_recall, expected_precision_recall)

    def test_f1_score(self):
        # Test case to check if the F1 score is calculated correctly
        expected_f1_score = {0: 0.5, 1: 0.6666666666666666, 2: 0}
        self.assertEqual(self.evaluator.f1_score, expected_f1_score)


if __name__ == "__main__":
    unittest.main()
