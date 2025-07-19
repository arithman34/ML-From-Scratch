import unittest
from src.supervised_learning.linear_model import LinearRegression, LogisticRegression, \
    LassoRegression, RidgeRegression
from src.data.data_generator import get_classification_data, get_regression_data
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, \
    LogisticRegression as SklearnLogisticRegression, Lasso as SklearnLasso, Ridge as SklearnRidge
from sklearn.metrics import accuracy_score
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

    def test_fit_and_predict(self):
        np.random.seed(42)
        reg = LinearRegression(max_iter=1000, learning_rate=0.01)
        reg.fit(self.X_train, self.y_train)

        sklearn_reg = SklearnLinearRegression()
        sklearn_reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)
        y_pred_sklearn = sklearn_reg.predict(self.X_test)

        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error
        mae_sklearn = np.mean(abs(y_pred_sklearn - self.y_test))

        abs_error = np.abs(mae_custom - mae_sklearn)

        self.assertLessEqual(abs_error, 0.05, f"MAE mismatch too high got {mae_custom:.4f}, but {mae_sklearn:.4f} for base")


class TestLassoRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data()

    def test_initialization(self):
        reg = LassoRegression(alpha=0.1, max_iter=1, learning_rate=0.01)

        self.assertIsNone(reg.W, "Weights should be initialized to None")
        self.assertIsNone(reg.B, "Bias should be initialized to None")
        self.assertEqual(reg.alpha, 0.1, "Alpha should be set correctly")
        self.assertEqual(reg.learning_rate, 0.01, "Learning rate should be set correctly")
        self.assertEqual(reg.max_iter, 1, "Max iterations should be set correctly")

    def test_fit_and_predict(self):
        np.random.seed(42)
        reg = LassoRegression(alpha=0.1, max_iter=1000, learning_rate=0.01)
        reg.fit(self.X_train, self.y_train)

        sklearn_reg = SklearnLasso(alpha=0.1, max_iter=1000)
        sklearn_reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)
        y_pred_sklearn = sklearn_reg.predict(self.X_test)

        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error
        mae_sklearn = np.mean(abs(y_pred_sklearn - self.y_test))

        abs_error = np.abs(mae_custom - mae_sklearn)

        self.assertLessEqual(abs_error, 0.05, f"MAE mismatch too high got {mae_custom:.4f}, but {mae_sklearn:.4f} for base")


class TestRidgeRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data()

    def test_initialization(self):
        reg = RidgeRegression(alpha=0.1, max_iter=1, learning_rate=0.01)

        self.assertIsNone(reg.W, "Weights should be initialized to None")
        self.assertIsNone(reg.B, "Bias should be initialized to None")
        self.assertEqual(reg.alpha, 0.1, "Alpha should be set correctly")
        self.assertEqual(reg.learning_rate, 0.01, "Learning rate should be set correctly")
        self.assertEqual(reg.max_iter, 1, "Max iterations should be set correctly")

    def test_fit_and_predict(self):
        np.random.seed(42)
        reg = RidgeRegression(alpha=0.1, max_iter=1000, learning_rate=0.01)
        reg.fit(self.X_train, self.y_train)

        sklearn_reg = SklearnRidge(alpha=0.1, max_iter=1000)
        sklearn_reg.fit(self.X_train, self.y_train)

        y_pred = reg.predict(self.X_test)
        y_pred_sklearn = sklearn_reg.predict(self.X_test)

        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error
        mae_sklearn = np.mean(abs(y_pred_sklearn - self.y_test))
        
        abs_error = np.abs(mae_custom - mae_sklearn)

        self.assertLessEqual(abs_error, 0.05, f"MAE mismatch too high got {mae_custom:.4f}, but {mae_sklearn:.4f} for base")


class TestLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data(num_samples=10000, num_classes=2)

    def test_initialization(self):
        clf = LogisticRegression(max_iter=1, learning_rate=0.01)

        self.assertIsNone(clf.W, msg="Weights should be initialized to None")
        self.assertIsNone(clf.B, msg="Bias should be initialized to None")
        self.assertEqual(clf.learning_rate, 0.01, msg="Learning rate should be set correctly")
        self.assertEqual(clf.max_iter, 1, msg="Max iterations should be set correctly")

    def test_fit_and_predict(self):
        np.random.seed(42)
        clf = LogisticRegression(max_iter=1000, learning_rate=0.01)
        clf.fit(self.X_train, self.y_train)

        sklearn_clf = SklearnLogisticRegression(penalty=None, max_iter=1000)
        sklearn_clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)
        y_pred_sklearn = sklearn_clf.predict(self.X_test)

        # Check accuracy
        acc_custom = accuracy_score(self.y_test, y_pred)
        acc_sklearn = accuracy_score(self.y_test, y_pred_sklearn)

        abs_error = np.abs(acc_custom - acc_sklearn)

        print(f"Custom Logistic Regression Accuracy: {acc_custom:.4f}"
              f", Sklearn Logistic Regression Accuracy: {acc_sklearn:.4f}")

        self.assertLessEqual(abs_error, 0.05, f"Accuracy mismatch too high got {acc_custom:.4f} and {acc_sklearn:.4f} for base")


if __name__ == '__main__':
    unittest.main()
