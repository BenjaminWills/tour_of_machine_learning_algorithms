from machine_learning_algorithms.bayesian_algorithms.naive_bayes.naive_bayes import (
    naive_bayes_classifier,
    find_class_count,
)
import numpy as np

from typing import Dict, Union


def calculate_gaussian_probability(x: float, mean: float, std: float) -> float:
    """
    Calculate the Gaussian probability of a given value.

    Parameters
    ----------
    x : float
        The value for which to calculate the Gaussian probability.
    mean : float
        The mean of the Gaussian distribution.
    std : float
        The std of the Gaussian distribution.

    Returns
    -------
    float
        The Gaussian probability of the given value.
    """
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(
        -((x - mean) ** 2) / (2 * std**2)
    )


def calculate_gaussian_mean_and_std(
    independent_variable: np.ndarray, classification_column: np.ndarray
) -> Dict[Union[int, str], Dict[str, float]]:
    """
    Calculate the mean and std of the Gaussian distribution for each class.

    Parameters
    ----------
    independent_variable : np.ndarray
        The independent variable column of the dataset.
    classification_column : np.ndarray
        The classification column of the dataset.

    Returns
    -------
    Dict[Union[int, str], Dict[str, float]]
        The mean and std of the Gaussian distribution for each class.
    """
    unique_classes = np.unique(classification_column)

    mean_and_std = {}

    for unique_class in unique_classes:
        # Get the mean and variance of the independent variable for each class
        classification_indices = classification_column == unique_class
        mean = np.mean(independent_variable[classification_indices])
        std = np.std(independent_variable[classification_indices])

        mean_and_std[unique_class] = {"mean": mean, "std": std}

    return mean_and_std


class gaussian_naive_bayes(naive_bayes_classifier):
    def __init__(
        self,
        indepdendent_variables: np.ndarray,
        classification_column: np.ndarray,
        laplace_smoothing_parameter: float = 0.001,
    ) -> None:
        super().__init__(
            indepdendent_variables, classification_column, laplace_smoothing_parameter
        )

    def calculate_posterior_probabilities(
        self,
        independent_variables: np.ndarray,
    ) -> Dict[Union[int, str], Dict[Union[int, str], float]]:
        """Calculate the gaussian posterior probabilities of each class given the independent variable.
        Thus each value within the output is the probability of seeing a class given that
        we know the independent variable.

        The outputs look like:
        {
            indep_var:
            {
                classification_var_1: count_1,
                ...,
                classification_var_n: count_n
            },
        ...}
        Parameters
        ----------
        independent_variables : np.ndarray
            The independent variable column of the dataset.

        Returns
        -------
        Dict[Union[int, str], float]: The posterior probabilities of each class given the independent variable.

        """

        """
        So, we need to calculate the posterior probabilities of each class given the independent variable.

        thus we loop through each independent variable column and calculate the following:
            * the mean and variance of the independent variable for each class in the form:
                {
                    indep_var: {
                        classification_var_1: {
                            mean: mean_1,
                            variance: variance_1
                        },
                        ...,
                        classification_var_n: {
                            mean: mean_n,
                            variance: variance_n
                        }
                    },
                    }
                }
            * the gaussian probability of each class given the independent variable in the form:
                {
                    indep_var: {
                        classification_var_1: probability_1,
                        ...,
                        classification_var_n: probability_n
                    }
        """
