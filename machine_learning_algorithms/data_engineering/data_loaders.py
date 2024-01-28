import pandas as pd
import numpy as np

from machine_learning_algorithms.data_engineering.one_hot_encoder import one_hot_encode

from typing import Tuple


def load_data(csv_path: str, dependent_column_name: str) -> Tuple[np.ndarray]:
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


def load_categorical_data(
    csv_path: str, classification_column_name: str
) -> Tuple[np.ndarray]:
    """
    Load data from a CSV file and separate the classification column and independent variables.

    Args:
        csv_path (str): The path to the CSV file.
        classification_column_name (str): The name of the classification column.

    Returns:
        np.ndarray: The classification column array.
        np.ndarray: The independent variables array.
    """
    dataframe = pd.read_csv(csv_path, index_col=None)

    dataframe, one_hot_encoded_categorical_column = one_hot_encode(
        dataframe, classification_column_name
    )

    dependent_variables = dataframe.to_numpy()
    categorical_data = one_hot_encoded_categorical_column.to_numpy()

    return independent_variables, categorical_data
