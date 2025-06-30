import numpy as np


class GaussianAnomalyDetector:
    def __init__(self, threshold):
        self.threshold = threshold
        self.mean = None
        self.variance = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.variance = np.var(X, axis=0)

    def gaussian_pdf(self, X):  # Gaussian Probability Density Function
        coefficient  = 1 / (np.sqrt(2 * np.pi * self.variance))
        exp_term = np.exp(-(X - self.mean) ** 2 / (2 * self.variance))

        return np.prod(coefficient  * exp_term, axis=1)

    def predict(self, X):        
        p = self.gaussian_pdf(X)
        return p < self.threshold
