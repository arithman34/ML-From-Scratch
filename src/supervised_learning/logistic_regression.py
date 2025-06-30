import numpy as np


class Sigmoid:
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def backward(z: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid function."""
        sig = Sigmoid.forward(z)
        return sig * (1 - sig)


class LogisticRegression:
    def __init__(self, max_iter=1000, learning_rate=0.001):
        self.W = None  # Weights
        self.B = None  # Bias
        self.J = None  # Cost history
        self.learning_rate = learning_rate  # Learning rate
        self.max_iter = max_iter  # Max iterations for convergence

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        m, n = X.shape  # m is the number of samples, n is the number of features

        # Initialize the weights and bias
        self.W = np.zeros(n)
        self.B = 0.0

        history = {}

        # Gradient descent
        for _ in range(self.max_iter):
            # Compute the prediction
            f_wb = self.predict_proba(X)

            # Compute the cost
            loss = -np.mean(y * np.log(f_wb) + (1-y) * np.log(1 - f_wb))  # Compute the loss

            # Compute the gradients
            dw = (1 / m) * np.dot(X.T, (f_wb - y))
            db = (1 / m) * np.sum(f_wb - y)

            # Update the parameters
            self.W -= self.learning_rate * dw
            self.B -= self.learning_rate * db

            history.setdefault('loss', []).append(loss)

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return Sigmoid.forward(np.dot(X, self.W) + self.B)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        
        return (probabilities >= 0.5).astype(int).flatten()
