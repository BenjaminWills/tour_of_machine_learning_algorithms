import numpy as np
from typing import Callable

from tqdm import tqdm


def calculate_gradient(
    function: Callable, point: np.ndarray, dx: float = 0.0001
) -> np.ndarray:
    """
    Calculate the gradient of a function at a given point.

    Args:
        function (Callable): Function to calculate the gradient of.
        point (np.ndarray): Point at which to calculate the gradient.
        dx (float, optional): Small change in the input to calculate the gradient. Defaults to 0.0001.

    Returns:
        np.ndarray: Gradient of the function at the given point.
    """
    # Calculate the gradient
    gradient = np.zeros(point.shape)
    dimension = point.shape[0]

    for i in range(dimension):
        # Take a small step in the ith direction
        step = np.zeros(point.shape)
        step[i] = dx

        # Calculate the gradient in the ith direction
        gradient[i] = (function(point + step) - function(point - step)) / (2 * dx)

    return gradient


def gradient_descent(
    function: Callable,
    initial_point: np.ndarray,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
    iteration_display_frequency: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform gradient descent to find the minimum of a function.

    Args:
        function (Callable): Function to find the minimum of.
        initial_point (np.ndarray): Initial point to start the gradient descent from.
        learning_rate (float, optional): Learning rate to use in the gradient descent. Defaults to 0.01.
        tolerance (float, optional): Tolerance to use for the convergence criterion. Defaults to 1e-6.
        max_iterations (int, optional): Maximum number of iterations to use. Defaults to 10000.

    Returns:
        np.ndarray: Minimum of the function.
    """
    # Set the initial point
    point = initial_point

    # Perform gradient descent
    for i in tqdm(range(max_iterations)):
        # Calculate the gradient at the current point
        gradient = calculate_gradient(function, point)

        # Print the progress at every 100 iterations
        if i % iteration_display_frequency == 0 and verbose:
            print(f"Function value at ieteration {i}: {function(point)}")
            print(f"Gradient magnitude: {np.linalg.norm(gradient)}")

        # Check if the magnitude of the gradient is less than the tolerance
        if np.linalg.norm(gradient) < tolerance:
            break

        # Normalise gradient s.t it has unit length
        gradient /= np.linalg.norm(gradient)

        # Take a step in the opposite direction of the gradient
        new_point = point - learning_rate * gradient

        # Check if the new point is the same as the old point
        if np.allclose(new_point, point, atol=1e-6):
            break

        # Update the point
        point = new_point

    return point
