import numpy as np


def pca(data: np.ndarray, num_components: int) -> np.ndarray:
    covariance_matrix = np.cov(data)

    # check the determinant of the covariance matrix
    if np.linalg.det(covariance_matrix) == 0:
        raise ValueError("The covariance matrix is not invertible.")

    # Check if number of components is greater than the number of features
    if num_components > data.shape[1]:
        raise ValueError(
            "The number of components cannot be greater than the number of features."
        )

    # Check if number of components is greater than the number of samples
    if num_components > data.shape[0]:
        raise ValueError(
            "The number of components cannot be greater than the number of samples."
        )

    # Check argument types
    if not isinstance(data, np.ndarray):
        raise TypeError("Data should be a numpy array.")
    if not isinstance(num_components, int):
        raise TypeError("Number of components should be an integer.")

    # calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Make vectors into unit vectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    # sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # select the top k eigenvectors
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

    # project the data onto the top k eigenvectors
    projected_data = np.dot(data, top_eigenvectors)

    return projected_data
