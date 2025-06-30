import numpy as np


class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    @staticmethod
    def euclidean_distance(x1, x2):
        # Compute Euclidean distance between two points x1 and x2
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X, y):
        # NOTE - This is why KNN is called a lazy learner
        self.X = X
        self.y = y

    def predict(self, X):
        raise NotImplementedError


class KNeighborsClassifier(KNearestNeighbors):
    def predict_proba(self, X):
        # Compute the distances between each test point and all training points
        distances = np.array([self.euclidean_distance(x_test, self.X) for x_test in X])

        # Sort by ascending order for the first k indices
        k_indices = np.argsort(distances, axis=1)[:, :self.k]

        # Get the target for the k indices
        k_y_pred = self.y[k_indices]

        # Calculate probabilities (count occurrences of each class)
        num_classes = np.max(self.y) + 1
        class_counts = np.zeros((X.shape[0], num_classes))

        for i, y_pred in enumerate(k_y_pred):
            for label in y_pred:
                class_counts[i, label] += 1

        # Calculate probabilities by dividing by k
        class_probs = class_counts / self.k

        return class_probs

    def predict(self, X):
        # Get the class with the highest probability
        y_pred = np.array([np.argmax(probs) for probs in self.predict_proba(X)])
        return y_pred


class KNeighborsRegressor(KNearestNeighbors):
    def predict_single(self, x):
        # Compute the distances between the point and all training points
        distances = self.euclidean_distance(x, self.X)

        # Sort by ascending order for the first k indices
        k_indices = np.argsort(distances)[:self.k]

        # Get the target values for the k nearest neighbors
        k_y_pred = self.y[k_indices]

        # Return the mean of the k neighbors
        prediction = np.mean(k_y_pred)
        return prediction
    
    def predict(self, X):
        # Predict for each test point
        y_pred = np.array([self.predict_single(x) for x in X])
        return y_pred