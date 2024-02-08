import unittest
import numpy as np
from machine_learning_algorithms.bayesian_algorithms.naive_bayes.naive_bayes import (
    find_class_count,
    calculate_prior_probabilities,
    calculate_posterior_probabilities,
    naive_bayes_classifier,
)


class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.independent_variables = np.array(
            [["A", "C", "D"], ["B", "C", "E"], ["A", "C", "F"]]
        )
        self.classification_column = np.array([0, 1, 0])

    def test_find_class_count(self):
        # Test case to check if find_class_count returns the correct class counts
        expected_result = {0: 2, 1: 1}
        self.assertEqual(find_class_count(self.classification_column), expected_result)

    def test_calculate_prior_probabilities(self):
        # Test case to check if calculate_prior_probabilities returns the correct prior probabilities
        expected_result = {0: 0.6666666666666666, 1: 0.3333333333333333}
        self.assertEqual(
            calculate_prior_probabilities(self.classification_column), expected_result
        )

    def test_calculate_posterior_probabilities(self):
        # Test case to check if calculate_posterior_probabilities returns the correct posterior probabilities
        independent_variable_column = self.independent_variables[:, 0]
        expected_result = {
            "A": {0: 1, 1: 0.001},
            "B": {0: 0.001, 1: 1},
        }
        self.assertEqual(
            calculate_posterior_probabilities(
                independent_variable_column, self.classification_column
            ),
            expected_result,
        )

    def test_naive_bayes_classifier(self):
        # Test case to check if naive_bayes_classifier initializes correctly
        classifier = naive_bayes_classifier(
            self.independent_variables, self.classification_column
        )
        self.assertEqual((classifier.classes == np.array([0, 1])).all(), True)
        self.assertEqual(
            classifier.priors, {0: 0.6666666666666666, 1: 0.3333333333333333}
        )
        self.assertEqual(
            classifier.posteriors,
            {
                0: {"A": {0: 1.0, 1: 0.001}, "B": {1: 1.0, 0: 0.001}},
                1: {"C": {0: 1.0, 1: 1.0}},
                2: {
                    "D": {0: 0.5, 1: 0.001},
                    "E": {1: 1.0, 0: 0.001},
                    "F": {0: 0.5, 1: 0.001},
                },
            },
        )

    def test_calculate_class_probability(self):
        # Test case to check if calculate_class_probability returns the correct class probability
        classifier = naive_bayes_classifier(
            self.independent_variables, self.classification_column
        )
        # F is not seen in the data. Hence, the probability should be 0.001
        input_data = np.array(["A", "F", "D"])
        expected_class = 0
        expected_result = {0: 0.0003333333333333333, 1: 3.3333333333333337e-10}
        self.assertEqual(
            classifier.predict(input_data), (expected_class, expected_result)
        )


if __name__ == "__main__":
    unittest.main()
