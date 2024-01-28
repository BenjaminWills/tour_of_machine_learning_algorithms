# import unittest
# import numpy as np
# from machine_learning_algorithms.regression_algorithms.logistic_regression.logistic_regression import (
#     logistic_regressor,
# )


# class TestLogisticRegressor(unittest.TestCase):
#     def setUp(self):
#         # Create a sample dataset for testing
#         self.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
#         self.y_train = np.array([0, 0, 1, 1])
#         self.X_test = np.array([[5, 6], [6, 7]])

#         # Create an instance of the logistic regressor
#         self.lr = logistic_regressor(
#             data_path=None,
#             classification_variable_name=None,
#             threshold=0.5,
#             learning_rate=0.01,
#             number_of_iterations=10,
#             gradient_threshold=0.5,
#             iteration_display_frequency=100,
#         )

#     def test_predict(self):
#         # Test the predict method
#         predicted_classes = self.lr.predict(self.X_test)
#         self.assertEqual(predicted_classes.shape, (2,))
#         self.assertTrue(
#             np.all(np.logical_or(predicted_classes == 0, predicted_classes == 1))
#         )

#     def test_find_optimal_coefficients(self):
#         # Test the find_optimal_coefficients function
#         coefficients = self.lr.find_optimal_coefficients(
#             self.X_train,
#             self.y_train,
#             learning_rate=0.01,
#             number_of_iterations=1000,
#             gradient_threshold=0.5,
#             iteration_display_frequency=100,
#         )
#         self.assertEqual(coefficients.shape, (3,))


# if __name__ == "__main__":
#     unittest.main()
