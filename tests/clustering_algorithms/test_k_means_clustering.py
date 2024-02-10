# import unittest
# import numpy as np
# from machine_learning_algorithms.clustering_algorithms.k_means.k_means_clustering import (
#     k_means_clustering,
# )
# from sklearn.datasets import make_blobs

# X, y = make_blobs(n_samples=10000, centers=3, n_features=2, random_state=42)


# class TestKMeansClustering(unittest.TestCase):
#     def setUp(self):
#         self.data = X
#         self.k = 3
#         self.max_iterations = 100
#         self.clustering = k_means_clustering(self.data, self.k, self.max_iterations)

#     def test_pick_initial_centroids(self):
#         centroids = self.clustering.pick_initial_centroids()
#         self.assertEqual(len(centroids), self.k)
#         for centroid in centroids:
#             self.assertIn(centroid.tolist(), self.data.tolist())

#     def test_package_cluster_information(self):
#         cluster_dist_array = np.array(
#             [[0, 1.5], [1, 2.5], [0, 0.5], [1, 1.5], [0, 0.1]]
#         )
#         packaged_cluster_information = self.clustering.package_cluster_information(
#             cluster_dist_array
#         )
#         expected_output = {
#             0.0: [
#                 {
#                     "datapoint_value": np.array([-2.9688544, 7.93444368]),
#                     "datapoint_distance": 1.5,
#                 },
#                 {
#                     "datapoint_value": np.array([-1.89542677, 9.38974199]),
#                     "datapoint_distance": 0.5,
#                 },
#                 {
#                     "datapoint_value": np.array([5.34113448, 2.69624001]),
#                     "datapoint_distance": 0.1,
#                 },
#             ],
#             1.0: [
#                 {
#                     "datapoint_value": np.array([3.16120524, -0.25650696]),
#                     "datapoint_distance": 2.5,
#                 },
#                 {
#                     "datapoint_value": np.array([3.8035403, 0.10672153]),
#                     "datapoint_distance": 1.5,
#                 },
#             ],
#         }

#         self.assertEqual(expected_output, packaged_cluster_information)

#     def test_calculate_centroid_distances(self):
#         centroids = [np.array([1, 2]), np.array([3, 4])]
#         packaged_cluster_information = self.clustering.calculate_centroid_distances(
#             centroids
#         )
#         self.assertEqual(len(packaged_cluster_information), self.k)
#         self.assertEqual(len(packaged_cluster_information[0]), len(self.data))
#         self.assertEqual(len(packaged_cluster_information[1]), len(self.data))
#         self.assertEqual(
#             packaged_cluster_information[0][0]["datapoint_value"].tolist(), [1, 2]
#         )
#         self.assertAlmostEqual(
#             packaged_cluster_information[0][0]["datapoint_distance"], 0.0
#         )

#     def test_calculate_mean_centroids(self):
#         packaged_cluster_information = {
#             0: [
#                 {"datapoint_value": np.array([1, 2]), "datapoint_distance": 0.0},
#                 {"datapoint_value": np.array([5, 6]), "datapoint_distance": 0.5},
#                 {"datapoint_value": np.array([9, 10]), "datapoint_distance": 0.1},
#             ],
#             1: [
#                 {"datapoint_value": np.array([3, 4]), "datapoint_distance": 0.0},
#                 {"datapoint_value": np.array([7, 8]), "datapoint_distance": 0.5},
#             ],
#         }
#         new_centroids = self.clustering.calculate_mean_centroids(
#             packaged_cluster_information
#         )
#         self.assertEqual(len(new_centroids), self.k)
#         self.assertEqual(new_centroids[0].tolist(), [5.0, 6.0])
#         self.assertEqual(new_centroids[1].tolist(), [5.0, 6.0])

#     def test_stop_if_clusters_are_static(self):
#         new_centroids = [np.array([1, 2]), np.array([3, 4])]
#         old_centroids = [np.array([1, 2]), np.array([3, 4])]
#         self.assertTrue(
#             self.clustering.stop_if_clusters_are_static(new_centroids, old_centroids)
#         )

#         new_centroids = [np.array([1, 2]), np.array([3, 4])]
#         old_centroids = [np.array([1, 2]), np.array([5, 6])]
#         self.assertFalse(
#             self.clustering.stop_if_clusters_are_static(new_centroids, old_centroids)
#         )

#     def test_train(self):
#         self.assertEqual(len(self.clustering.centroids), self.k)
#         for centroid in self.clustering.centroids:
#             self.assertIn(centroid.tolist(), self.data.tolist())

#     def test_predict(self):
#         input_data = np.array([1, 2])
#         cluster_index = self.clustering.predict(input_data)
#         self.assertIn(cluster_index, [0, 1])


# if __name__ == "__main__":
#     unittest.main()
