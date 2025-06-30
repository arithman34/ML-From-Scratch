import numpy as np


class LinearRegression:
    def __init__(self, max_iter=1000, learning_rate=0.01):
        self.W = None  # Weights
        self.B = None  # Bias
        self.learning_rate = learning_rate  # Learning rate
        self.max_iter = max_iter  # Max iterations for convergence

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape  # m is the number of samples, n is the number of features

        # Initialize the weights and bias
        self.W = np.zeros(n)
        self.B = 0.0

        history = {}

        # Gradient descent
        for _ in range(self.max_iter):
            # Compute the prediction
            f_wb = np.dot(X, self.W) + self.B

            # Compute the cost
            cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)  # TODO - Use the loss instead of the cost

            # Compute the gradients
            dW = (1 / m) * np.dot(X.T, (f_wb - y))
            dB = (1 / m) * np.sum(f_wb - y)

            # Update the parameters
            self.W -= self.learning_rate * dW
            self.B -= self.learning_rate * dB

            history.setdefault('cost', []).append(cost)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.W) + self.B