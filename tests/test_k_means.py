import unittest
import numpy as np
from sklearn.metrics import adjusted_rand_score
from src.unsupervised_learning.k_means import KMeans
from tests.test_data import get_clustering_data


class TestKMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = 3  # Number of clusters
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_clustering_data()

    def test_initialization(self):
        kmeans = KMeans(k=self.k, max_iter=1, tol=1e-4)
        kmeans.fit(self.X_train)

        self.assertEqual(kmeans.centroids.shape, (self.k, 2), "Centroids should have the correct shape")

    def test_find_closest_centroids(self):
        kmeans = KMeans(k=self.k, max_iter=1, tol=1e-4)
        kmeans.fit(self.X_train)

        idx = kmeans.find_closest_centroids(self.X_test)
        self.assertTrue(np.all((idx >= 0) & (idx < self.k)), "Cluster indices should be within valid range")

    def test_compute_centroids(self):
        kmeans = KMeans(k=self.k, max_iter=1, tol=1e-4)
        kmeans.fit(self.X_train)

        idx = kmeans.find_closest_centroids(self.X_train)
        new_centroids = kmeans.compute_centroids(self.X_train, idx)
        self.assertEqual(new_centroids.shape, (self.k, 2), "New centroids should have the correct shape")

    def test_cost_function(self):
        kmeans = KMeans(k=self.k, max_iter=1, tol=1e-4)
        cost = kmeans.fit(self.X_train)

        self.assertGreaterEqual(cost, 0, "Cost function should be non-negative")

    def test_predict(self):
        kmeans = KMeans(k=self.k, max_iter=300, tol=1e-4, random_state=42)
        kmeans.fit(self.X_train)

        y_pred = kmeans.predict(self.X_test)

        accuracy = adjusted_rand_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.9, msg=f"Similarity should be higher, got {accuracy:.2f}")


if __name__ == "__main__":
    unittest.main()
