import pandas as pd
import numpy as np

from machine_learning_algorithms.logger import make_logger
from machine_learning_algorithms.loss_functions.mean_squared_error import (
    mean_squared_error,
)

logger = make_logger()


def load_data(csv_path: str, dependent_column_name: str) -> np.ndarray:
    """
    Load data from a CSV file and separate the dependent variable and independent variables.

    Args:
        csv_path (str): The path to the CSV file.
        dependent_column_name (str): The name of the dependent variable column.

    Returns:
        np.ndarray: The dependent variable array.
        np.ndarray: The independent variables array.
    """
    dataframe = pd.read_csv(csv_path, index_col=None)

    dependent_variable = dataframe[dependent_column_name].to_numpy()
    independent_variables = dataframe.drop([dependent_column_name], axis=1).to_numpy()

    return dependent_variable, independent_variables


def find_optimal_coefficients(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """
    Find the optimal coefficients for the ordinary least squares regression model.

    Args:
        x_data (np.ndarray): The independent variables.
        y_data (np.ndarray): The dependent variable.

    Returns:
        np.ndarray: The optimal coefficients.
    """
    n = len(x_data)

    # Define the matrix A with a one in the first column to account for the intercept.
    A = np.c_[np.ones(n), x_data]

    # Calculate the coefficients.
    pseudo_inverse = np.linalg.inv(np.dot(A.T, A))

    # intercept, coefficient_1, coefficient_2, ...
    co_efficients = np.dot(pseudo_inverse, np.dot(A.T, y_data))

    intercept = co_efficients[0]
    co_efficients_to_print = co_efficients[1:]

    logger.info(
        f"Regression intercept: {intercept}, regression coefficients: {co_efficients_to_print}"
    )

    return co_efficients


class olsr_regressor:
    def __init__(self, csv_path: str, dependent_variable_name: str) -> None:
        """
        Initialize the OLSR regressor.

        Args:
            csv_path (str): The path to the CSV file.
            dependent_variable_name (str): The name of the dependent variable column.
        """
        self.dependent_variable_name = dependent_variable_name

        dependent_variable, independent_variables = load_data(
            csv_path, dependent_variable_name
        )

        self.optimal_coefficients = find_optimal_coefficients(
            independent_variables, dependent_variable
        )

        self.dependent_variable = dependent_variable
        self.independent_variable = independent_variables

    def predict(self, independent_variables: np.array) -> float:
        """
        Predict the dependent variable value based on the given independent variables.

        Args:
            independent_variables (np.array): The independent variables.

        Returns:
            float: The predicted dependent variable value.
        """
        intercept = self.optimal_coefficients[0]
        coefficents = self.optimal_coefficients[1:]

        prediction = np.dot(coefficents, independent_variables)

        return intercept + prediction

    def make_multiple_predictions(self, test_path: str):
        """
        Make multiple predictions on the test data and calculate the mean square error.

        Args:
            test_path (str): The path to the test data CSV file.

        Returns:
            float: The mean square error.
            list: The predicted values.
            list: The true values.
        """
        dependent_variable, independent_variables = load_data(
            test_path, self.dependent_variable_name
        )
        predictions = list(map(self.predict, independent_variables))

        mean_square_error = mean_squared_error(
            np.array(predictions), dependent_variable
        )

        logger.info(
            f"\t\n Error when predicting on the test path: {test_path} \t\n The mean square error is {mean_square_error}"
        )

        return mean_square_error, predictions, dependent_variable
