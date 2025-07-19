import numpy as np


class KMeans:
    def __init__(self, k, max_iter=100, tol=1e-4, random_state=None):
        self.k = k  # Number of clusters to find
        self.max_iter = max_iter  # Max iterations
        self.centroids = None  # Centroids
        self.tol = tol  # Tolerance for convergence

        if random_state is not None:
            np.random.seed(random_state)

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def init_centroids(self, X):
        # Get the number of samples and the number of features
        m, _ = X.shape

        initial_indices = np.random.choice(m, self.k, replace=False)  # Avoid using the same point multiple times
        centroids = X[initial_indices]

        return centroids

    def find_closest_centroids(self, X):
        m, _ = X.shape
        idx = np.zeros(m, dtype=int)

        for i in range(m):
            min_dist = float('inf')

            for j in range(self.k):
                dist = self.euclidean_distance(X[i], self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    idx[i] = j

        return idx
    
    def compute_centroids(self, X, idx):
        _, n = X.shape

        centroids = np.zeros((self.k, n))

        for k in range(self.k):
            indices = np.where(idx == k)
            if len(indices[0]) > 0:  # Avoid division by zero (will reduce number of clusters)
                centroids[k] = np.mean(X[indices], axis=0)

        return centroids
    
    def cost(self, X):
        idx = self.find_closest_centroids(X)
        closest_centroids = self.centroids[idx]

        squared_diffs = np.sum((X - closest_centroids) ** 2, axis=1)
        J = np.sum(squared_diffs)

        return J

    def fit(self, X):
        self.centroids = self.init_centroids(X)

        for iteration in range(self.max_iter):
            idx = self.find_closest_centroids(X)

            new_centroids = self.compute_centroids(X, idx)
            centroid_diff = np.linalg.norm(new_centroids - self.centroids, axis=None)
            self.centroids = new_centroids

            if centroid_diff < self.tol:
                break
            
        return self.cost(X)

    def predict(self, X):
        return self.find_closest_centroids(X)
    

# Implement DBSCAN