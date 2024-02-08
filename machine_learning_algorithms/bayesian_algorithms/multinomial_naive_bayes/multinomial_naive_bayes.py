from machine_learning_algorithms.bayesian_algorithms.naive_bayes.naive_bayes import (
    naive_bayes_classifier,
    find_class_count,
)

import numpy as np

from typing import Dict, Union

from collections import defaultdict


class multinomial_naive_bayes_classifier(naive_bayes_classifier):
    def __init__(
        self,
        indepdendent_variables: np.ndarray,
        classification_column: np.ndarray,
        laplace_smoothing_parameter: float = 0.001,
    ):
        super().__init__(
            indepdendent_variables, classification_column, laplace_smoothing_parameter
        )

    def calculate_posterior_probabilities(
        self,
        independent_variables: np.ndarray,
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
        independent_variables : np.ndarray
            The independent variable column of the dataset.

        Returns
        -------
        Dict[Union[int, str], float]: The posterior probabilities of each class given the independent variable.

        """
        unique_independent_variables = np.unique(independent_variables)
        unique_classes = np.unique(self.classification_column)

        # Unique class count
        unique_class_counts = find_class_count(self.classification_column)

        # Create a dictionary to store the counts of each class for each unique independent variable
        conditional_probability_stroage = {}

        for independent_variable in unique_independent_variables:
            # Find indices of where the independent variable is equal to the unique independent variable
            independent_indices = np.where(
                independent_variables == independent_variable
            )

            # Find the counts of the classes within each independent variable value
            classes = self.classification_column[independent_indices]
            class_counts = find_class_count(classes)

            # Unseen classes are given a probability of alpha
            unseen_classes = set(unique_classes) - set(class_counts.keys())

            # Calculate the conditional probability of each class given the independent variable
            for class_ in unique_classes:
                # Set the count to alpha if the class is not seen
                if class_ in unseen_classes:
                    class_counts[class_] = self.laplace_smoothing_parameter
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
        for (
            independent_variable,
            class_counts,
        ) in conditional_probability_stroage.items():
            # class_counts looks like {class: proportion, ...} we need the proportions for each class to sum to 1
            # across each independent variable e.g {1: {0:0.5,1:0.5}, 2: {0:0.5,1:0.5}}
            for class_, proportion in class_counts.items():
                validation_dict[class_] += proportion
        for class_, proportion in validation_dict.items():
            assert (
                abs(proportion - 1) < 0.05
            ), f"Proportions for class {class_} do not sum to 1, they sum to {proportion} \n {dict(validation_dict)} \n {dict(conditional_probability_stroage)}"
        return conditional_probability_stroage
