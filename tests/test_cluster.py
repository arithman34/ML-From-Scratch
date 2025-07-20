import unittest
import numpy as np
from src.unsupervised_learning.cluster import KMeans, DBSCAN
from src.data.data_generator import get_clustering_data, get_moon_data
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans as SKLearnKMeans, DBSCAN as SKLearnDBSCAN


class TestKMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = 3  # Number of clusters
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_clustering_data(centers=cls.k, num_samples=300, random_state=42)

    def test_initialization(self):
        k_means = KMeans(k=self.k, max_iter=1, tol=1e-4)
        k_means.fit(self.X_train)

        self.assertEqual(k_means.centroids.shape, (self.k, 2), "Centroids should have the correct shape")

    def test_find_closest_centroids(self):
        k_means = KMeans(k=self.k, max_iter=1, tol=1e-4)
        k_means.fit(self.X_train)

        idx = k_means.find_closest_centroids(self.X_test)
        self.assertTrue(np.all((idx >= 0) & (idx < self.k)), "Cluster indices should be within valid range")

    def test_compute_centroids(self):
        k_means = KMeans(k=self.k, max_iter=1, tol=1e-4)
        k_means.fit(self.X_train)

        idx = k_means.find_closest_centroids(self.X_train)
        new_centroids = k_means.compute_centroids(self.X_train, idx)
        self.assertEqual(new_centroids.shape, (self.k, 2), "New centroids should have the correct shape")

    def test_cost_function(self):
        k_means = KMeans(k=self.k, max_iter=1, tol=1e-4)
        cost = k_means.fit(self.X_train)

        self.assertGreaterEqual(cost, 0, "Cost function should be non-negative")

    def test_fit_and_predict(self):
        k_means = KMeans(k=self.k, max_iter=300, tol=1e-4, random_state=42)
        k_means.fit(self.X_train)

        sklearn_k_means = SKLearnKMeans(n_clusters=self.k, max_iter=300, tol=1e-4, random_state=42)
        sklearn_k_means.fit(self.X_train)

        y_pred = k_means.predict(self.X_test)
        y_pred_sklearn = sklearn_k_means.predict(self.X_test)

        accuracy = adjusted_rand_score(self.y_test, y_pred)
        accuracy_sklearn = adjusted_rand_score(self.y_test, y_pred_sklearn)

        abs_error = np.abs(accuracy - accuracy_sklearn)

        self.assertLessEqual(abs_error, 0.05, f"Adjusted Rand Index mismatch too high got {accuracy:.4f} and {accuracy_sklearn:.4f} for base")


class TestDBSCAN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eps = 0.5
        cls.min_samples = 5
        cls.X, _, cls.y, _ = get_moon_data(num_samples=300, noise=None, random_state=42)

    def test_fit_and_predict(self):
        db_scan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db_scan.fit_predict(self.X)

        sklearn_db_scan = SKLearnDBSCAN(eps=self.eps, min_samples=self.min_samples)
        sklearn_labels = sklearn_db_scan.fit_predict(self.X)

        accuracy = adjusted_rand_score(self.y, labels)
        accuracy_sklearn = adjusted_rand_score(self.y, sklearn_labels)

        abs_error = np.abs(accuracy - accuracy_sklearn)

        self.assertLessEqual(abs_error, 0.05, f"Adjusted Rand Index mismatch too high got {accuracy:.4f} and {accuracy_sklearn:.4f} for base")


if __name__ == "__main__":
    unittest.main()
