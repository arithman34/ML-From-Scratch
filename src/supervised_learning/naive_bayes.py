import numpy as np


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = None
        self.feature_log_prob = None
        self.classes = None

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_log_prior = np.log(counts / counts.sum())

        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.feature_log_prob = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            class_feature_count = X_cls.sum(axis=0)
            smoothed_fc = class_feature_count + self.alpha
            smoothed_total = smoothed_fc.sum()
            self.feature_log_prob[i] = np.log(smoothed_fc / smoothed_total)

    def predict(self, X):
        log_probs = X @ self.feature_log_prob.T + self.class_log_prior
        return self.classes[np.argmax(log_probs, axis=1)]


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.class_log_prior = None
        self.feature_mean = None
        self.feature_var = None
        self.classes = None
        self.epsilon = None  # Will store the smoothing value to be added to variances

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_log_prior = np.log(counts / counts.sum())

        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.feature_mean = np.zeros((n_classes, n_features))
        self.feature_var = np.zeros((n_classes, n_features))

        # Compute the mean and variance for each class
        for i, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.feature_mean[i] = X_cls.mean(axis=0)
            self.feature_var[i] = X_cls.var(axis=0)

        # Compute epsilon for smoothing: var_smoothing * max(var of all features)
        self.epsilon = self.var_smoothing * np.var(X, axis=0).max()
        self.feature_var += self.epsilon

    def predict(self, X):
        log_probs = []
        for i in range(len(self.classes)):
            mean = self.feature_mean[i]
            var = self.feature_var[i]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_probs.append(log_prob + self.class_log_prior[i])
        
        log_probs = np.array(log_probs).T
        return self.classes[np.argmax(log_probs, axis=1)]
