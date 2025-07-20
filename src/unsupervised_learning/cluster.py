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
        return np.sqrt(np.sum((x1 - x2) ** 2))

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
    

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
        self.min_samples = min_samples  # Number of samples in a neighborhood for a point to be considered as a core point
        self.labels_ = None  # Cluster labels for each point
        self.core_sample_indices_ = None  # Indices of core samples

    def fit(self, X):
        n_points = X.shape[0]
        self.labels_ = -np.ones(n_points, dtype=int)  # Initialize all points as noise (-1)
        self.core_sample_indices_ = []  # Reset core sample indices
        cluster_id = 0
        visited = np.zeros(n_points, dtype=bool)  # Track visited points

        # Loop through each point
        for point_idx in range(n_points):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._get_neighbors(X, point_idx)

            # If not enough neighbors, it's noise or border point
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1  # Mark as noise
            else:
                # This is a core point - add to core sample indices
                self.core_sample_indices_.append(point_idx)
                # Start a new cluster
                self._grow_cluster(X, point_idx, neighbors, cluster_id, visited)
                cluster_id += 1

    def fit_predict(self, X):
        """Fit the model and return cluster labels"""
        self.fit(X)
        return self.labels_

    def _get_neighbors(self, X, point_idx):
        """Return indices of all points within self.eps of X[point_idx]"""
        # Use euclidean distance to find neighbors
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))

        neighbor_indices = np.where(distances <= self.eps)[0]
        return neighbor_indices.tolist()

    def _grow_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        """Grow a cluster from a core point"""
        self.labels_[point_idx] = cluster_id
        i = 0

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._get_neighbors(X, neighbor_idx)

                # If neighbor is also a core point, expand the neighborhood
                if len(new_neighbors) >= self.min_samples:
                    # Add to core sample indices if not already there
                    if neighbor_idx not in self.core_sample_indices_:
                        self.core_sample_indices_.append(neighbor_idx)
                    
                    for n in new_neighbors:
                        if n not in neighbors:
                            neighbors.append(n)

            # Assign to cluster if it hasn't been assigned yet
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

            i += 1
    

# Implement DBSCAN