import numpy as np

from machine_learning_algorithms.optimisation_algorithms.gradient_descent.gradient_descent import (
    gradient_descent,
)


class SVM:
    def __init__(
        self, independent_variables: np.ndarray, classification_variables: np.ndarray
    ) -> None:
        """
        Initialize the SVM classifier.

        Parameters:
        - independent_variables: numpy.ndarray
            The independent variables used for classification.
        - classification_variables: numpy.ndarray
            The classification variables, which should be +1 or -1.

        Returns:
        None
        """
        # Input variables
        self.independent_variables = independent_variables

        # Classification variables should be +1 or -1.
        self.classification_variables = classification_variables

        # Find lagrange multipliers
        lagrange_multipliers = self.optimise_dual_problem()

        # Calculate the weight vector
        self.weights = self.calculate_weights(lagrange_multipliers)

        # Calculate the bias term
        self.bias = self.calculate_bias(self.weights, lagrange_multipliers)

    def calculate_weights(self, lagrange_multipliers: np.ndarray) -> np.ndarray:
        """
        Calculate the weight vector based on the lagrange multipliers.

        Parameters:
        - lagrange_multipliers: numpy.ndarray
            The lagrange multipliers obtained from the optimization.

        Returns:
        numpy.ndarray
            The weight vector.
        """
        rows, cols = self.independent_variables.shape
        weights: np.ndarray = np.zeros(cols)
        for row_index in range(rows):
            weights += (
                lagrange_multipliers[row_index]
                * self.classification_variables[row_index]
                * self.independent_variables[row_index]
            )
        return weights

    def calculate_bias(
        self, weights: np.ndarray, lagrange_multipliers: np.ndarray
    ) -> float:
        """
        Calculate the bias term based on the weights and lagrange multipliers.

        Parameters:
        - weights: numpy.ndarray
            The weight vector.
        - lagrange_multipliers: numpy.ndarray
            The lagrange multipliers obtained from the optimization.

        Returns:
        float
            The bias term.
        """
        bias = 0
        for index in range(len(lagrange_multipliers)):
            bias += self.classification_variables[index] - np.dot(
                weights, self.independent_variables[index]
            )
        return bias / len(lagrange_multipliers)

    def dual_problem_objective_function(self, lagrange_multipliers: np.ndarray):
        """
        Objective function for the dual problem optimization.

        Parameters:
        - lagrange_multipliers: numpy.ndarray
            The lagrange multipliers.

        Returns:
        float
            The value of the objective function.
        """
        weights = self.calculate_weights(lagrange_multipliers)
        return sum(lagrange_multipliers) - 0.5 * np.linalg.dot(weights, weights)

    def optimise_dual_problem(self) -> np.ndarray:
        """
        Optimize the dual problem using gradient descent.

        Returns:
        numpy.ndarray
            The optimized lagrange multipliers.
        """
        initial_lagrange_multipliers = np.zeros(len(self.classification_variables))
        return gradient_descent(
            self.dual_problem_objective_function, initial_lagrange_multipliers
        )

    def predict(self, independent_variables: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the given independent variables.

        Parameters:
        - independent_variables: numpy.ndarray
            The independent variables for prediction.

        Returns:
        numpy.ndarray
            The predicted class labels.
        """
        return np.sign(np.dot(independent_variables, self.weights) + self.bias)
