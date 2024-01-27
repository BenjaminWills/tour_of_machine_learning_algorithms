import pandas as pd
import numpy as np

from machine_learning_algorithms.logger import make_logger

logger = make_logger()


def load_data(csv_path: str, dependent_column_name: str) -> np.ndarray:
    dataframe = pd.read_csv(csv_path, index_col=None)

    dependent_variable = dataframe[dependent_column_name].to_numpy()
    independent_variables = dataframe.drop([dependent_column_name], axis=1).to_numpy()

    return dependent_variable, independent_variables


def mean_square_error(
    true_answers: np.ndarray, coefficients: np.ndarray, dependent_variables: np.ndarray
) -> float:
    return np.linalg.norm(true_answers - np.dot(dependent_variables, coefficients)) ** 2


def find_optimal_coefficients(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    n = len(x_data)

    # Define the matrix A with a one in the first column to account for the intercept.
    A = np.c_[np.ones(n), x_data]

    # Calculate teh coefficents.
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
        intercept = self.optimal_coefficients[0]
        coefficents = self.optimal_coefficients[1:]

        prediction = np.dot(coefficents, independent_variables)

        return intercept + prediction

    def make_multiple_predictions(self, test_path: str):
        dependent_variable, independent_variables = load_data(
            test_path, self.dependent_variable_name
        )
        errors = []
        predictions = []

        for dependent, independent in zip(dependent_variable, independent_variables):
            prediction = self.predict(independent)
            error = (dependent - prediction) ** 2
            errors.append(error)
            predictions.append(prediction)

        mean_square_error = np.array(errors).mean()

        logger.info(
            f"\t\n Error when predicting on the test path: {test_path} \t\n The mean square error is {mean_square_error}"
        )

        return mean_square_error, predictions, dependent_variable
