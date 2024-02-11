import numpy as np

from typing import Dict


class pca:
    def __init__(self, data: np.ndarray) -> None:
        # Check argument types
        if not isinstance(data, np.ndarray):
            raise TypeError("Data should be a numpy array.")
        self.data = data

        self.pca = self.get_pca()
        self.order_of_importance = np.argsort(self.pca["eigenvalues"])[::-1]

    def get_pca(self) -> Dict[str, np.ndarray]:
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
