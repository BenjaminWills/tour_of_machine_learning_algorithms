import numpy as np

from typing import List, Dict

from collections import defaultdict
from tqdm import tqdm

CLUSTER_INDEX = int
DATAPOINT_VALUE = np.ndarray
DATAPOINT_DISTANCE = float

PACKAGED_CLUSTER_INFORMATION = Dict[
    CLUSTER_INDEX, List[Dict[str, DATAPOINT_VALUE | DATAPOINT_DISTANCE]]
]


def calculate_euclidian_distance(point_1: np.ndarray, point_2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        point_1: The first point.
        point_2: The second point.

    Returns:
        The Euclidean distance between the two points.
    """
    return np.linalg.norm(point_1 - point_2)


def calculate_inertia(
    packaged_cluster_information: PACKAGED_CLUSTER_INFORMATION,
) -> Dict[int, float]:
    """
    Calculate the inertia for each cluster.

    Inertia is defined as the sum of distances between each data point and its centroid within a cluster.

    Args:
        packaged_cluster_information: A dictionary containing the cluster information.

    Returns:
        A dictionary where the keys are cluster indices and the values are the corresponding inertia values.
    """
    inertia_dict = {}
    for (
        cluster_index,
        list_of_datapoint_information,
    ) in packaged_cluster_information.items():
        inertia = sum(
            [
                datapoint_information["datapoint_distance"]
                for datapoint_information in list_of_datapoint_information
            ]
        )
        inertia_dict[cluster_index] = inertia
    return inertia_dict


def calculate_dunn_index(
    packaged_cluster_information: PACKAGED_CLUSTER_INFORMATION,
    centroids: List[np.array],
) -> float:
    """
    Calculate the Dunn index for the clustering.

    The Dunn index is a measure of cluster separation and compactness.

    Args:
        packaged_cluster_information: A dictionary containing the cluster information.
        centroids: A list of centroid arrays.

    Returns:
        The Dunn index for the clustering.
    """
    # Calculate the distance between centroids
    cluster_distances = []
    for i, centroid_1 in enumerate(centroids):
        for j, centroid_2 in enumerate(centroids):
            if i > j:
                cluster_distance = calculate_euclidian_distance(centroid_1, centroid_2)
                cluster_distances.append(cluster_distance)

    # Calculate the distance within clusters
    cluster_diameters = []
    for (
        cluster_index,
        list_of_datapoint_information,
    ) in packaged_cluster_information.items():
        cluster_diameter = max(
            [
                datapoint_information["datapoint_distance"]
                for datapoint_information in list_of_datapoint_information
            ]
        )
        cluster_diameters.append(cluster_diameter)

    # Calculate the Dunn index
    max_intracluster_distance = max(cluster_diameters)
    min_intercluster_distance = min(cluster_distances)

    dunn_index = min_intercluster_distance / max_intracluster_distance
    return dunn_index


class k_means_clustering:
    def __init__(
        self, independent_variables: np.ndarray, k: int, max_iterations: int = 10_000
    ) -> None:
        """
        Initialize the k-means clustering algorithm.

        Args:
            independent_variables: The independent variables (data points) to be clustered.
            k: The number of clusters.
            max_iterations: The maximum number of iterations for the algorithm.
        """
        self.independent_variables = independent_variables

        self.num_data_points = len(independent_variables)

        if k > self.num_data_points:
            raise ValueError(
                f"k ({k}) must be smaller than the number of datapoints ({self.num_data_points})"
            )
        self.k = k
        self.max_iterations = max_iterations

        # Define the clusters
        self.centroids = self.train()

    def pick_initial_centroids(self) -> List[np.array]:
        """
        Pick initial centroids randomly from the data points.

        Returns:
            A list of initial centroid arrays.
        """
        # Generate k random indices
        centroid_indices = np.random.uniform(0, self.num_data_points, self.k).astype(
            int
        )
        centroids: List[np.ndarray] = [
            self.independent_variables[centroid_index, :]
            for centroid_index in centroid_indices
        ]
        return centroids

    def package_cluster_information(
        self,
        cluster_dist_array: np.ndarray,
    ) -> PACKAGED_CLUSTER_INFORMATION:
        """
        Package the cluster information into a readable dictionary.

        Args:
            cluster_dist_array: An array containing the cluster indices and distances.

        Returns:
            A dictionary containing the cluster information.
        """
        packaged_cluster_information = defaultdict(list)
        for datapoint_index, cluster_dist in enumerate(cluster_dist_array):
            # Unpack row value
            cluster_index, datapoint_distance = cluster_dist

            datapoint_information = {
                "datapoint_value": self.independent_variables[datapoint_index],
                "datapoint_distance": datapoint_distance,
            }

            # Fill in dictionary
            packaged_cluster_information[cluster_index].append(datapoint_information)

        return packaged_cluster_information

    def calculate_centroid_distances(
        self, centroids: List[np.array]
    ) -> PACKAGED_CLUSTER_INFORMATION:
        """
        Calculate the distances between data points and centroids.

        Args:
            centroids: A list of centroid arrays.

        Returns:
            A dictionary containing the cluster information.
        """
        # Define a matrix of points and the distances from their centroids
        # Such that the (i,j) entry is the distance of the i'th datapoint
        # from the j'th centroid.
        centroid_distances = np.zeros((self.num_data_points, self.k))

        for datapoint_index, datapoint in enumerate(self.independent_variables):
            for centroid_index, centroid in enumerate(centroids):
                centroid_distances[
                    datapoint_index, centroid_index
                ] = calculate_euclidian_distance(datapoint, centroid)

        # Find the smallest index in each row and return the index of the
        # cluster that it should belong to.
        cluster_indices = np.argmin(centroid_distances, axis=1)
        smallest_distances = np.min(centroid_distances, axis=1)

        # Package these variables up into a readable dictionary
        cluster_dist_array = np.c_[cluster_indices, smallest_distances]
        packaged_cluster_information = self.package_cluster_information(
            cluster_dist_array
        )
        return packaged_cluster_information

    def calculate_mean_centroids(
        self, packaged_cluster_information: PACKAGED_CLUSTER_INFORMATION
    ):
        """
        Calculate the mean centroids based on the cluster information.

        Args:
            packaged_cluster_information: A dictionary containing the cluster information.

        Returns:
            A list of new centroid arrays.
        """
        new_centroids = []
        # Extract the distances from the cluster information
        for (
            cluster_id,
            list_of_datapoint_information,
        ) in packaged_cluster_information.items():
            datapoint_values = np.stack(
                [
                    datapoint_information["datapoint_value"]
                    for datapoint_information in list_of_datapoint_information
                ]
            )

            new_centroid = np.mean(datapoint_values.T, axis=1)
            new_centroids.append(new_centroid)
        return new_centroids

    def stop_if_clusters_are_static(
        self,
        new_centroids: List[np.array],
        old_centroids: List[np.array],
    ):
        """
        Check if the clusters have become static (centroids have not changed).

        Args:
            new_centroids: The new centroid arrays.
            old_centroids: The old centroid arrays.

        Returns:
            True if the clusters have become static, False otherwise.
        """
        for new_centroid, old_centroid in zip(new_centroids, old_centroids):
            if np.all(new_centroid == old_centroid):
                return True
        return False

    def train(self) -> List[np.ndarray]:
        """
        Train the k-means clustering algorithm.

        Returns:
            A list of final centroid arrays.
        """
        # Initialise centroids
        initial_centroids = self.pick_initial_centroids()

        centroids = initial_centroids

        for iteration in tqdm(range(self.max_iterations)):
            # Define which points belong to each cluster
            cluster_information = self.calculate_centroid_distances(centroids)

            # Define the new centroids
            new_centroids = self.calculate_mean_centroids(cluster_information)

            if iteration % 1_000 == 0:
                inertia = calculate_inertia(cluster_information)
                dunn_index = calculate_dunn_index(cluster_information, new_centroids)
                print(f"Inertia: {inertia}\nDunn Index: {dunn_index}")

            if self.stop_if_clusters_are_static(new_centroids, centroids):
                # End loop if the centroids have not changed
                print("CLUSTERS ARE STATIC")
                break

            centroids = new_centroids

        self.inertia = calculate_inertia(cluster_information)
        self.dunn_index = calculate_dunn_index(cluster_information, centroids)

        return centroids

    def predict(self, input_data: np.ndarray) -> CLUSTER_INDEX:
        """
        Predict the cluster index for a given input data point.

        Args:
            input_data: The input data point.

        Returns:
            The cluster index that the input data point belongs to.
        """
        # Calculate the distance from the input data to each cluster,
        # then calculate the smallest argument and return it.
        distances = [
            calculate_euclidian_distance(input_data, centroids)
            for centroids in self.centroids
        ]

        centriod = np.argmin(distances)

        return centriod
