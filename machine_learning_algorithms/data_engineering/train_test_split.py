import numpy as np


def test_train_split(data: np.ndarray, train_size: float = 0.8, random_seed: int = 42):
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Shuffle the inputted data
    shuffled_data = np.random.shuffle(data)

    # Calculate number in the training set
    training_rows = int(len(data) * train_size)
    testing_rows = len(data) - training_rows

    # Split the data
    training_data = data[:training_rows]
    testing_data = data[training_rows:]

    train_X, train_y = training_data[:, :-1], training_data[:, -1]
    test_X, test_y = testing_data[:, :-1], testing_data[:, -1]

    return {
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y,
    }
