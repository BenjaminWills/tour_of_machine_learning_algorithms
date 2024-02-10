from machine_learning_algorithms.clustering_algorithms.k_means.k_means_clustering import (
    k_means_clustering,
    PACKAGED_CLUSTER_INFORMATION,
)
import numpy as np


class k_medians_clustering(k_means_clustering):
    def __init__(
        self, independent_variables: np.ndarray, k: int, max_iterations: int = 10000
    ) -> None:
        super().__init__(independent_variables, k, max_iterations)

    # Override the calculate_mean_centroids method
    def calculate_mean_centroids(
        self, packaged_cluster_information: PACKAGED_CLUSTER_INFORMATION
    ):
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

            new_centroid = np.median(datapoint_values.T, axis=1)
            new_centroids.append(new_centroid)
        return new_centroids
