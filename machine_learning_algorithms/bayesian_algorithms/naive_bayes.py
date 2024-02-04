import numpy as np

from typing import Dict, Union

from collections import Counter


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
    num_classes = len(classification_column)
    class_counts = Counter(classification_column).most_common()
    return {class_: count / num_classes for class_, count in class_counts}


def calculate_posterior_probabilities() -> None:
    pass
