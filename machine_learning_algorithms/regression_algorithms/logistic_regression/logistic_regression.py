import numpy as np

from machine_learning_algorithms.loss_functions.cross_entropy import cross_entropy

from machine_learning_algorithms.optimisation_algorithms.gradient_descent.gradient_descent import (
    gradient_descent,
)

from machine_learning_algorithms.loss_functions.sigmoid import sigmoid


def find_optimal_coefficients(
    independent_variables: np.ndarray,
    classification_column: np.ndarray,
    learning_rate: float,
    number_of_iterations: int,
    gradient_threshold: float,
    iteration_display_frequency: int,
) -> np.ndarray:
    """
    Find the optimal coefficients for the logistic regression model.

    Returns:
        np.ndarray: The optimal coefficients.
    """
    number_of_samples, number_of_features = independent_variables.shape

    # Real data is a matrix with the shape (number of samples, number of features) and we want to add a column of ones
    # to the matrix to represent the intercept term
    independent_variables = np.c_[np.ones(number_of_samples), independent_variables]

    # Initialize the coefficients to zero
    initial_coefficients = np.zeros(number_of_features + 1)

    # Define our cost function that we wish to minimise
    def cost_function(coefficients: np.ndarray) -> float:
        # A list of the form [[0.5,0.5], [0.2, 0.8]]
        predicted_probabilities = []
        for independent_variable in independent_variables:
            probability_of_class_1 = sigmoid(np.dot(independent_variable, coefficients))
            probability_of_class_0 = 1 - probability_of_class_1
            predicted_probabilities.append(
                np.array([probability_of_class_0, probability_of_class_1])
            )

        # Expected output classes
        # [[1, 0], [0, 1]]
        # We assume that the classification columns are one-hot encoded
        cost = cross_entropy(predicted_probabilities, classification_column)
        return cost

    # minimise the cost function w.r.t the coefficients
    coefficients = gradient_descent(
        function=cost_function,
        initial_point=initial_coefficients,
        learning_rate=learning_rate,
        tolerance=gradient_threshold,
        max_iterations=number_of_iterations,
        iteration_display_frequency=iteration_display_frequency,
    )

    return coefficients


class logistic_regressor:
    def __init__(
        self,
        independent_variables: np.ndarray,
        classification_variable: np.ndarray,
        learning_rate: float = 0.01,
        number_of_iterations: int = 1000,
        gradient_threshold: float = 0.5,
        iteration_display_frequency: int = 100,
    ) -> None:
        self.coefficients = find_optimal_coefficients(
            independent_variables,
            classification_variable,
            learning_rate,
            number_of_iterations,
            gradient_threshold,
            iteration_display_frequency,
        )

    def predict(
        self, independent_variables: np.ndarray, probability_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict the classes of the independent variables.

        Args:
            independent_variables (np.ndarray): The independent variables.
            probability_threshold (float): The probability threshold to use to determine the class.

        Returns:
            np.ndarray: The predicted classes.
        """
        number_of_samples, number_of_features = independent_variables.shape

        # Real data is a matrix with the shape (number of samples, number of features) and we want to add a column of ones
        # to the matrix to represent the intercept term
        independent_variables = np.c_[np.ones(number_of_samples), independent_variables]

        predicted_probabilities = []
        for independent_variable in independent_variables:
            probability_of_class_1 = sigmoid(
                np.dot(independent_variable, self.coefficients)
            )
            probability_of_class_0 = 1 - probability_of_class_1
            predicted_probabilities.append(
                np.array([probability_of_class_0, probability_of_class_1])
            )

        predicted_probabilities = np.array(predicted_probabilities)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)
        predicted_classes = np.where(
            predicted_probabilities[:, 1] > probability_threshold, 1, 0
        )
        return predicted_classes
