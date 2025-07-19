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
            cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)

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
    

class LassoRegression(LinearRegression):
    def __init__(self, alpha=1.0, max_iter=1000, learning_rate=0.01):
        super().__init__(max_iter, learning_rate)
        self.alpha = alpha  # Regularization strength

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape

        # Initialize the weights and bias
        self.W = np.zeros(n)
        self.B = 0.0

        history = {}

        # Gradient descent with L1 regularization
        for _ in range(self.max_iter):
            f_wb = np.dot(X, self.W) + self.B
            cost = (1 / (2 * m)) * (np.sum((f_wb - y) ** 2) + self.alpha * np.sum(np.abs(self.W)))

            dW = (1 / m) * (np.dot(X.T, (f_wb - y)) + self.alpha * np.sign(self.W))
            dB = (1 / m) * np.sum(f_wb - y)

            self.W -= self.learning_rate * dW
            self.B -= self.learning_rate * dB

            history.setdefault('cost', []).append(cost)

        return history
    

class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0, max_iter=1000, learning_rate=0.01):
        super().__init__(max_iter, learning_rate)
        self.alpha = alpha  # Regularization strength

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape

        # Initialize the weights and bias
        self.W = np.zeros(n)
        self.B = 0.0

        history = {}

        # Gradient descent with L2 regularization
        for _ in range(self.max_iter):
            f_wb = np.dot(X, self.W) + self.B
            cost = (1 / (2 * m)) * (np.sum((f_wb - y) ** 2) + self.alpha * np.sum(self.W ** 2))

            dW = (1 / m) * (np.dot(X.T, (f_wb - y)) + self.alpha * self.W)
            dB = (1 / m) * np.sum(f_wb - y)

            self.W -= self.learning_rate * dW
            self.B -= self.learning_rate * dB

            history.setdefault('cost', []).append(cost)

        return history
    

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
