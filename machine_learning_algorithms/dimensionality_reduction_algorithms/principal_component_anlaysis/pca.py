import numpy as np

from typing import Dict


class pca:
    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize the PCA class.

        Args:
            data (np.ndarray): The input data as a numpy array.

        Raises:
            TypeError: If the data is not a numpy array.

        """
        # Check argument types
        if not isinstance(data, np.ndarray):
            raise TypeError("Data should be a numpy array.")
        self.data = data

        self.pca = self.get_pca()
        self.order_of_importance = np.argsort(self.pca["eigenvalues"])[::-1]

    def get_pca(self) -> Dict[str, np.ndarray]:
        """
        Perform Principal Component Analysis (PCA) on the input data.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the eigenvalues and eigenvectors.

        """
        # Calcualte the covariance matrix of the independent variables
        covariance_matrix = np.cov(self.data.T)

        # calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Make vectors into unit vectors
        eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }

    def reduce_dimensionality(self, num_components: int) -> np.ndarray:
        """
        Reduce the dimensionality of the input data using PCA.

        Args:
            num_components (int): The number of components to keep.

        Returns:
            np.ndarray: The projected data with reduced dimensionality.

        Raises:
            ValueError: If the number of components is greater than the number of features or samples.
            TypeError: If the number of components is not an integer.

        """
        # Check if number of components is greater than the number of features
        if num_components > self.data.shape[1]:
            raise ValueError(
                "The number of components cannot be greater than the number of features."
            )

        # Check if number of components is greater than the number of samples
        if num_components > self.data.shape[0]:
            raise ValueError(
                "The number of components cannot be greater than the number of samples."
            )

        if not isinstance(num_components, int):
            raise TypeError("Number of components should be an integer.")

        eigenvectors = self.pca["eigenvectors"]

        # select the top k eigenvectors
        top_eigenvectors = eigenvectors[:, self.order_of_importance[:num_components]]

        # project the data onto the top k eigenvectors
        projected_data = np.dot(self.data, top_eigenvectors)

        return projected_data
