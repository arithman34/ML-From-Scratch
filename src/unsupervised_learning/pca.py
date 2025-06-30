from typing import Optional
import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_

        # Covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False, bias=True)

        # Eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.eigenvectors_ = eigenvectors[:, sorted_indices]

        # Select the n_components eigenvectors
        self.components_ = self.eigenvectors_[:, :self.n_components].T

        # Select the n_components eigenvalues
        self.explained_variance_ratio_ = self.eigenvalues_[:self.n_components] / np.sum(self.eigenvalues_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_

        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
