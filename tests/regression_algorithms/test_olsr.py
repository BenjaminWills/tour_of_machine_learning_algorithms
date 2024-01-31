import unittest
import numpy as np

from machine_learning_algorithms.regression_algorithms.ordinary_least_squares.olsr import (
    find_optimal_coefficients,
    olsr_regressor,
)

from sklearn.datasets import load_diabetes

regression = load_diabetes()
independent_variables = regression.data
dependent_variable = regression.target


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
    def setUp(self):
        self.olsr = olsr_regressor(dependent_variable, independent_variables)

    def test_predict(self):
        independent_variables = np.array(
            [
                0.03807591,
                0.05068012,
                0.06169621,
                0.02187239,
                -0.0442235,
                -0.03482076,
                -0.04340085,
                -0.00259226,
                0.01990749,
                -0.01764613,
            ]
        )

        prediction = self.olsr.predict(independent_variables)

        # Assert that the prediction is of type float
        self.assertIsInstance(prediction, float)

        # Assert that the prediction has the correct value
        self.assertAlmostEqual(prediction, 206.1163854542022)

    def test_make_multiple_predictions(self):
        (
            mean_square_error,
            predictions,
            _,
        ) = self.olsr.make_multiple_predictions(
            independent_variables, dependent_variable
        )

        # Assert that the mean square error is of type float
        self.assertIsInstance(mean_square_error, float)

        # Assert that the predictions and dependent variable are of type of a numpy array
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(dependent_variable, np.ndarray)

        # Assert that the predictions and dependent variable have the correct lengths
        self.assertEqual(len(predictions), 442)
        self.assertEqual(len(dependent_variable), 442)


if __name__ == "__main__":
    unittest.main()
