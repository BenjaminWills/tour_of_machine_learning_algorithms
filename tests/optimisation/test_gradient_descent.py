import unittest
import numpy as np
from machine_learning_algorithms.optimisation_algorithms.gradient_descent.gradient_descent import (
    gradient_descent,
)


class TestGradientDescent(unittest.TestCase):
    def test_minimum_of_quadratic_function(self):
        # Test case to check if gradient_descent finds the minimum of a quadratic function
        def quadratic_function(x):
            return x**2

        initial_point = np.array([10.0])
        minimum = gradient_descent(quadratic_function, initial_point)
        self.assertAlmostEqual(minimum[0], 0.0, places=4)

    def test_minimum_of_rosenbrock_function(self):
        # Test case to check if gradient_descent finds the minimum of the Rosenbrock function
        a = 0

        def rosenbrock_function(x):
            return (a - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        initial_point = np.array([5.0, 12.0])
        minimum = gradient_descent(
            rosenbrock_function, initial_point, max_iterations=100_000
        )
        self.assertAlmostEqual(minimum[0], a, delta=0.1)
        self.assertAlmostEqual(minimum[1], a**2, delta=0.1)

    def test_minimum_of_custom_function(self):
        # Test case to check if gradient_descent finds the minimum of a custom function
        def custom_function(x):
            return np.sin(x) + np.cos(x)

        initial_point = np.array([0.0])
        minimum = gradient_descent(
            custom_function, initial_point, learning_rate=0.1, max_iterations=100_000
        )
        self.assertAlmostEqual(minimum[0], -3 * np.pi / 4, delta=0.05)


if __name__ == "__main__":
    unittest.main()
