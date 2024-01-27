import unittest
import numpy as np

from machine_learning_algorithms.regression_algorithms.ordinary_least_squares.olsr import (
    load_data,
    find_optimal_coefficients,
    olsr_regressor,
)


class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        csv_path = "test.csv"
        dependent_column_name = "target"

        dependent_variable, independent_variables = load_data(
            csv_path, dependent_column_name
        )

        # Assert that the dependent variable and independent variables are of type np.ndarray
        self.assertIsInstance(dependent_variable, np.ndarray)
        self.assertIsInstance(independent_variables, np.ndarray)

        # Assert that the dependent variable and independent variables have the correct shapes
        self.assertEqual(dependent_variable.shape, (3,))
        self.assertEqual(independent_variables.shape, (3, 2))


class TestFindOptimalCoefficients(unittest.TestCase):
    def test_find_optimal_coefficients(self):
        x_data = np.array([[5, 9], [12, 4], [5, 6]])
        y_data = np.array([7, 8, 9])

        optimal_coefficients = find_optimal_coefficients(x_data, y_data)

        # Assert that the optimal coefficients are of type np.ndarray
        self.assertIsInstance(optimal_coefficients, np.ndarray)

        # Assert that the optimal coefficients have the correct shape
        self.assertEqual(optimal_coefficients.shape, (3,))

        # Assert that the optimal coefficients have the correct values
        np.testing.assert_array_almost_equal(
            optimal_coefficients, np.array([14.666667, -0.333333, -0.666667])
        )


class TestOLSRRegressor(unittest.TestCase):
    def test_predict(self):
        csv_path = "test.csv"
        dependent_variable_name = "target"
        olsr = olsr_regressor(csv_path, dependent_variable_name)

        independent_variables = np.array([2, 3])

        prediction = olsr.predict(independent_variables)

        # Assert that the prediction is of type float
        self.assertIsInstance(prediction, float)

        # Assert that the prediction has the correct value
        self.assertAlmostEqual(prediction, 6)

    def test_make_multiple_predictions(self):
        csv_path = "test.csv"
        dependent_variable_name = "target"
        olsr = olsr_regressor(csv_path, dependent_variable_name)

        test_path = "test.csv"

        (
            mean_square_error,
            predictions,
            dependent_variable,
        ) = olsr.make_multiple_predictions(test_path)

        # Assert that the mean square error is of type float
        self.assertIsInstance(mean_square_error, float)

        # Assert that the predictions and dependent variable are of type of a numpy array
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(dependent_variable, np.ndarray)

        # Assert that the predictions and dependent variable have the correct lengths
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(dependent_variable), 3)


if __name__ == "__main__":
    unittest.main()
