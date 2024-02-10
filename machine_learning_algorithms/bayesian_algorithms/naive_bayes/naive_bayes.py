import numpy as np

from typing import Dict, Union

from collections import Counter


def find_class_count(
    classification_column: np.ndarray,
) -> Dict[Union[int, str], float]:
    """
    Count the occurrences of each class in the classification column.

    Parameters
    ----------
    classification_column : np.ndarray
        The classification column of the dataset.

    Returns
    -------
    Dict[Union[int, str], float]
        A dictionary containing the count of each class in the classification column.
    """
    class_counts = Counter(list(classification_column)).most_common()
    return {class_: count for class_, count in class_counts}


def calculate_prior_probabilities(
    classification_column: np.ndarray,
) -> Dict[Union[int, str], float]:
    """Calculate the prior probabilities of each class in the classification column.

    Parameters
    ----------
    classification_column : np.ndarray
        The classification column of the dataset.

    Returns
    -------
    Dict[Union[int, str], float]
        The prior probabilities of each class in the classification column.
    """
    num_rows = len(classification_column)
    return {
        class_: count / num_rows
        for class_, count in find_class_count(classification_column).items()
    }


class naive_bayes_classifier:
    def __init__(
        self,
        indepdendent_variables: np.ndarray,
        classification_column: np.ndarray,
        laplace_smoothing_parameter: float = 0.001,
    ) -> None:
        """
        Initialize the Naive Bayes Classifier.

        Parameters
        ----------
        indepdendent_variables : np.ndarray
            The independent variable column of the dataset.
        classification_column : np.ndarray
            The classification column of the dataset.
        """
        self.independent_variables = indepdendent_variables
        self.classification_column = classification_column
        self.laplace_smoothing_parameter = laplace_smoothing_parameter

        self.classes = np.unique(classification_column)

        self.priors = calculate_prior_probabilities(classification_column)
        self.posteriors = self.calculate_posteriors()

    def calculate_posteriors(
        self,
    ) -> Dict[Union[int, str], Dict[Union[int, str], float]]:
        """Calculates the posterior probabilities of each class given and independent variable in the training set.

        Returns
        -------
        Dict[Union[int, str], Dict[Union[int, str], float]]
            The posterior probabilities of each class given an independent variable.
        """
        posteriors = {}
        # Find the dimensionality of the independent variables
        num_independent_variables = self.independent_variables.shape[1]

        # Loop through each independent variable and calculate the posterior probabilities
        for index in range(num_independent_variables):
            independent_variables = self.independent_variables[:, index]
            posteriors[index] = self.calculate_posterior_probabilities(
                independent_variables
            )
        return posteriors

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
        probability = self.priors[class_]
        for independent_variable_index, independent_variable_value in enumerate(
            input_data
        ):
            try:
                # The probability of the specified class occurring given that we have a known
                # value of the independent variable P(y = y | X = x).
                posterior_probabilities = self.posteriors[independent_variable_index]
                variable_posterior_probabilities = posterior_probabilities[
                    independent_variable_value
                ]
                class_posterior_probability = variable_posterior_probabilities[class_]
                probability *= class_posterior_probability
            except KeyError as e:
                # This is the case when the independent variable is not seen in the training set.
                probability *= self.laplace_smoothing_parameter

        return probability

    def predict(self, input_data: np.ndarray) -> Union[int, str]:
        """
        Predict the class label for the input data.

        Parameters
        ----------
        input_data : np.ndarray
            The input data for which the class label is predicted.

        Returns
        -------
        Union[int, str]
            The predicted class label for the input data.
        """
        # We loop through each potential class and calculate the probabilities of seeing this.
        class_probabilities = {}
        for class_ in self.classes:
            class_probabilities[class_] = self.calculate_class_probability(
                input_data, class_
            )
        class_prediction = max(class_probabilities, key=class_probabilities.get)
        return class_prediction, class_probabilities
