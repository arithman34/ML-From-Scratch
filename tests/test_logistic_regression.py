import unittest
from src.supervised_learning.logistic_regression import LogisticRegression
from tests.test_data import get_classification_data
from sklearn.metrics import accuracy_score
import numpy as np


class TestLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data()

    def test_initialization(self):
        clf = LogisticRegression(max_iter=1, learning_rate=0.01)

        self.assertIsNone(clf.W, msg="Weights should be initialized to None")
        self.assertIsNone(clf.B, msg="Bias should be initialized to None")
        self.assertEqual(clf.learning_rate, 0.01, msg="Learning rate should be set correctly")
        self.assertEqual(clf.max_iter, 1, msg="Max iterations should be set correctly")

    def test_fit(self):
        clf = LogisticRegression(max_iter=1000, learning_rate=0.01)
        history = clf.fit(self.X_train, self.y_train)

        self.assertIsNotNone(clf.W, msg="Weights should be set after fitting")
        self.assertIsNotNone(clf.B, msg="Bias should be set after fitting")

        self.assertLessEqual(history['loss'][-1], history['loss'][0], "Final loss should be less than initial loss")

        # Sliding window size
        window_size = clf.max_iter // 5

        # Check loss within the sliding window
        for i in range(0, clf.max_iter, window_size):
            window_start = i
            window_end = i + window_size
            window_losses = history['loss'][window_start:window_end]

            # Ensure the last loss in the window is less than the first loss in the window
            self.assertGreater(window_losses[0], window_losses[-1], msg=f"Loss did not decrease in the window [{window_start}, {window_end}]")

    def test_predict(self):
        clf = LogisticRegression(max_iter=10000, learning_rate=0.01)
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)

        # Check accuracy
        acc_custom = accuracy_score(self.y_test, y_pred)

        self.assertGreaterEqual(acc_custom, 0.85, f"Accuracy should be higher, got {acc_custom:.2f}")

        # NOTE - Cannot check weights and bias as they could be anything


if __name__ == '__main__':
    unittest.main()
