import unittest
import numpy as np
from machine_learning_algorithms.instance_based_algorithms.support_vector_machine.svm import (
    SVM,
)

from sklearn.datasets import load_iris


regression = load_iris()
X, y = load_iris(return_X_y=True)

np.random.seed(42)

# Extract the first 100 samples (as they're the binary labels)
X = X[:100]
y = y[:100]

# Shuffle X and y with a random seed of 42
np.random.shuffle(X)
np.random.shuffle(y)

# Relabel 0's as -1's.
y = np.where(y == 0, -1, 1)

# Split the data into training and testing sets
train_y = y[:80]
train_X = X[:80]
test_y = y[80:]
test_X = X[80:]


class TestSVMClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.svm = SVM(train_X, train_y)
