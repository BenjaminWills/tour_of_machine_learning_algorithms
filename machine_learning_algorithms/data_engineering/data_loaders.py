import pandas as pd
import numpy as np


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
