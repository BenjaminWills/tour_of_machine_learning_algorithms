import numpy as np

from typing import Dict, Union

from collections import Counter, defaultdict


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


def calculate_posterior_probabilities(
    independent_variable_column: np.ndarray,
    classification_column: np.ndarray,
    alpha: float = 0.001,
) -> Dict[Union[int, str], Dict[Union[int, str], float]]:
    """Calculate the posterior probabilities of each class given the independent variable.
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
    independent_variable_column : np.ndarray
        The independent variable column of the dataset.
    classification_column : np.ndarray
        The classification column of the dataset.
    alpha : float, optional
        The Laplace smoothing parameter, by default 1.

    Returns
    -------
    Dict[Union[int, str], float]: The posterior probabilities of each class given the independent variable.

    """
    unique_independent_variables = np.unique(independent_variable_column)
    unique_classes = np.unique(classification_column)

    # Unique class count
    unique_class_counts = find_class_count(classification_column)

    # Create a dictionary to store the counts of each class for each unique independent variable
    conditional_probability_stroage = {}

    for independent_variable in unique_independent_variables:
        # Find indices of where the independent variable is equal to the unique independent variable
        independent_indices = np.where(
            independent_variable_column == independent_variable
        )

        # Find the counts of the classes within each independent variable value
        classes = classification_column[independent_indices]
        class_counts = find_class_count(classes)

        # Unseen classes are given a probability of alpha
        unseen_classes = set(unique_classes) - set(class_counts.keys())

        # Calculate the conditional probability of each class given the independent variable
        for class_ in unique_classes:
            # Set the count to alpha if the class is not seen
            if class_ in unseen_classes:
                class_counts[class_] = alpha
                # Skip the next if statement
                continue
            # Divide the count by the unique class count to find the proportion
            if class_ in class_counts:
                class_counts[class_] = (
                    class_counts[class_] / unique_class_counts[class_]
                )

        # Store the conditional probability of each class given the independent variable
        conditional_probability_stroage[independent_variable] = class_counts

    # Check that all probabilties of each class sum to 1
    validation_dict = defaultdict(lambda: 0)
    for independent_variable, class_counts in conditional_probability_stroage.items():
        # class_counts looks like {class: proportion, ...} we need the proportions for each class to sum to 1
        # across each independent variable e.g {1: {0:0.5,1:0.5}, 2: {0:0.5,1:0.5}}
        for class_, proportion in class_counts.items():
            validation_dict[class_] += proportion
    for class_, proportion in validation_dict.items():
        assert (
            abs(proportion - 1) < 0.05
        ), f"Proportions for class {class_} do not sum to 1, they sum to {proportion}"
    return conditional_probability_stroage


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
            independent_variable_column = self.independent_variables[:, index]
            posteriors[index] = calculate_posterior_probabilities(
                independent_variable_column,
                self.classification_column,
                alpha=self.laplace_smoothing_parameter,
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
