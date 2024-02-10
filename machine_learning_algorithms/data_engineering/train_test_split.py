import numpy as np


def test_train_split(data: np.ndarray, train_size: float = 0.8, random_seed: int = 42):
    """
    Splits the inputted data into training and testing sets.

    Parameters:
        data (np.ndarray): The input data to be split.
        train_size (float): The proportion of data to be used for training. Default is 0.8.
        random_seed (int): The random seed for reproducibility. Default is 42.

    Returns:
        dict: A dictionary containing the training and testing sets.
            - train_X (np.ndarray): The features of the training set.
            - train_y (np.ndarray): The labels of the training set.
            - test_X (np.ndarray): The features of the testing set.
            - test_y (np.ndarray): The labels of the testing set.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Shuffle the data
    shuffled_data = np.random.shuffle(data)

    # Split the data into training and testing sets
    training_rows = int(len(data) * train_size)

    # Split the data into training and testing sets
    training_data = shuffled_data[:training_rows]
    testing_data = shuffled_data[training_rows:]

    # Split the data into features and labels
    train_X, train_y = training_data[:, :-1], training_data[:, -1]
    test_X, test_y = testing_data[:, :-1], testing_data[:, -1]
    return {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y,
    }
