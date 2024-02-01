import unittest
import numpy as np
from machine_learning_algorithms.regression_algorithms.logistic_regression.logistic_regression import (
    logistic_regressor,
    find_optimal_coefficients,
)
from sklearn.datasets import load_iris

regression = load_iris()
independent_variable = regression.data
dependent_variable = regression.target

# Binary classes
final_binary_index = np.where(dependent_variable <= 1)[0][-1]

independent_variable = independent_variable[:final_binary_index]
dependent_variable = dependent_variable[:final_binary_index]

# One hot encode the data
one_hot_encoded_data = []
for class_ in dependent_variable:
    if class_ == 0:
        one_hot_encoded_data.append([1, 0])
    if class_ == 1:
        one_hot_encoded_data.append([0, 1])

one_hot_encoded_data = np.array(one_hot_encoded_data)

x_train = independent_variable[:80]
y_train = one_hot_encoded_data[:80]
x_test = independent_variable[80:]
y_test = one_hot_encoded_data[80:]


class TestLogisticRegressor(unittest.TestCase):
    def setUp(self):
        # Create an instance of the logistic regressor
        self.lr = logistic_regressor(
            x_train,
            y_train,
            learning_rate=0.01,
            number_of_iterations=10,
            gradient_threshold=0.5,
            iteration_display_frequency=100,
        )

    def test_predict(self):
        # Test the predict method
        predicted_classes = self.lr.predict(x_test, probability_threshold=0.5)
        self.assertEqual(predicted_classes.shape, (len(x_test),))
        self.assertTrue(
            np.all(np.logical_or(predicted_classes == 0, predicted_classes == 1))
        )

    def test_find_optimal_coefficients(self):
        # Test the find_optimal_coefficients function
        coefficients = find_optimal_coefficients(
            x_train,
            y_train,
            learning_rate=0.01,
            number_of_iterations=1000,
            gradient_threshold=0.5,
            iteration_display_frequency=100,
        )
        self.assertEqual(coefficients.shape, (5,))


if __name__ == "__main__":
    unittest.main()
