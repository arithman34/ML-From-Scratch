import unittest
from src.supervised_learning.linear_regression import LinearRegression
from src.data.data_generator import get_regression_data
import numpy as np


class TestLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data()

    def test_initialization(self):
        reg = LinearRegression(max_iter=1, learning_rate=0.01)

        self.assertIsNone(reg.W, "Weights should be initialized to None")
        self.assertIsNone(reg.B, "Bias should be initialized to None")
        self.assertEqual(reg.learning_rate, 0.01, "Learning rate should be set correctly")
        self.assertEqual(reg.max_iter, 1, "Max iterations should be set correctly")

    def test_fit(self):
        reg = LinearRegression(max_iter=1000, learning_rate=0.01)
        history = reg.fit(self.X_train, self.y_train)

        self.assertIsNotNone(reg.W, "Weights should be set after fitting")
        self.assertIsNotNone(reg.B, "Bias should be set after fitting")

        self.assertLessEqual(history['cost'][-1], history['cost'][0], "Final cost should be less than initial cost")

        # Sliding window size
        window_size = reg.max_iter // 5

        # Check loss within the sliding window
        for i in range(0, reg.max_iter, window_size):
            window_start = i
            window_end = i + window_size
            window_losses = history['cost'][window_start:window_end]

            # Ensure the last loss in the window is less than the first loss in the window
            self.assertGreater(window_losses[0], window_losses[-1], msg=f"Cost did not decrease in the window [{window_start}, {window_end}]")

    def test_predict(self):
        reg = LinearRegression(max_iter=1000, learning_rate=0.01)
        reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)

        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error

        self.assertLessEqual(mae_custom, 10, f"Mean Absolute Error should be lower, got {mae_custom:.2f}")


if __name__ == '__main__':
    unittest.main()
