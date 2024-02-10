from machine_learning_algorithms.bayesian_algorithms.naive_bayes.naive_bayes import (
    naive_bayes_classifier,
    find_class_count,
)
import numpy as np
from scipy.stats import norm


from typing import Dict, Union


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
    # INHERITENCE MAY NOT BE WISE...
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
        self.posteriors = self.calculate_posteriors()

    def calculate_posteriors(self):
        posterior_mean_std = {}
        for index, column in enumerate(self.independent_variables.T):
            posterior_mean_std[index] = calculate_gaussian_mean_and_std(
                column, self.classification_column
            )
        return posterior_mean_std

    def calculate_class_probability(
        self, input_data: np.ndarray, class_: Union[int, str]
    ) -> float:
        """
        Calculate the probability of a given class occurring given the input data.

        Parameters
        ----------
        input_data : np.ndarray
            The input data for which the class probability is calculated.
        class_ : Union[int, str]
            The class for which the probability is calculated.
        alpha : float, optional
            The Laplace smoothing parameter, by default 0.001.

        Returns
        -------
        float
            The probability of the specified class occurring given the input data.
        """
        prior_probability = self.priors[class_]
        probability = prior_probability
        for independent_variable_index, independent_variable_value in enumerate(
            input_data
        ):
            mean, std = self.posteriors[independent_variable_index][class_].values()
            # Calculate gaussian likelihood of seeing the class given the values that we've seen
            # P(y = y | X = x).
            if std != 0:
                probability *= norm.pdf(independent_variable_value, mean, std)
            else:
                probability *= norm.cdf(
                    independent_variable_value, mean, self.laplace_smoothing_parameter
                )
        return probability
