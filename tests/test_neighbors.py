import unittest
from src.supervised_learning.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier as SklearnKneighborsClassifier, KNeighborsRegressor as SklearnKNeighborsRegressor
from sklearn.metrics import accuracy_score
from src.data.data_generator import get_classification_data, get_regression_data
import numpy as np


class TestKNearestClassifer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data()

    def test_initialization(self):
        clf = KNeighborsClassifier(k=5)

        self.assertIsNone(clf.X, msg="X should be initialized to None")
        self.assertIsNone(clf.y, msg="y should be initialized to None")
        self.assertEqual(clf.k, 5, msg="k should be set correctly")

    def test_fit(self):
        clf = KNeighborsClassifier(k=5)
        clf.fit(self.X_train, self.y_train)

        self.assertIsNotNone(clf.X, msg="X should be set after fitting")
        self.assertIsNotNone(clf.y, msg="y should be set after fitting")

        np.testing.assert_array_equal(self.X_train.shape, clf.X.shape, "Shape of X_train and X should match")
        np.testing.assert_array_equal(self.y_train.shape, clf.y.shape, "Shape of y_train and y should match")

    def test_predict(self):
        # Fit custom KNeighborsClassifer
        clf = KNeighborsClassifier(k=5)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        # Fit sklearn KNeighborsClassifier
        sklearn_model = SklearnKneighborsClassifier(n_neighbors=5)
        sklearn_model.fit(self.X_train, self.y_train)
        sklearn_y_pred = sklearn_model.predict(self.X_test)

        # Accuracy
        acc_custom = accuracy_score(self.y_test, y_pred)
        acc_sklearn = accuracy_score(self.y_test, sklearn_y_pred)

        # Check mismatch ratio
        abs_error = np.abs(acc_custom - acc_sklearn)
        self.assertLessEqual(abs_error, 0.05, f"MAE mismatch too high got {acc_custom:.4f} and {acc_sklearn:.4f} for base")


class TestKNearestRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data()

    def test_initialization(self):
        reg = KNeighborsRegressor(k=5)

        self.assertIsNone(reg.X, msg="X should be initialized to None")
        self.assertIsNone(reg.y, msg="y should be initialized to None")
        self.assertEqual(reg.k, 5, msg="k should be set correctly")

    def test_fit(self):
        reg = KNeighborsRegressor(k=5)
        reg.fit(self.X_train, self.y_train)

        self.assertIsNotNone(reg.X, msg="X should be set after fitting")
        self.assertIsNotNone(reg.y, msg="y should be set after fitting")

        np.testing.assert_array_equal(self.X_train.shape, reg.X.shape, "Shape of X_train and X should match")
        np.testing.assert_array_equal(self.y_train.shape, reg.y.shape, "Shape of y_train and y should match")

    def test_predict(self):
        reg = KNeighborsRegressor(k=5)
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)

        # Fit sklearn KNeighborsRegressor
        sklearn_model = SklearnKNeighborsRegressor(n_neighbors=5)
        sklearn_model.fit(self.X_train, self.y_train)
        sklearn_y_pred = sklearn_model.predict(self.X_test)

        # Compare predictions
        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error (custom)
        mae_sklearn = np.mean(abs(sklearn_y_pred - self.y_test))  # Mean Absolute Error (sklearn)

        # Check mismatch ratio
        abs_mae = abs(mae_custom - mae_sklearn)
        self.assertLessEqual(abs_mae, 0.01, f"MAE mismatch too high got {mae_custom:.4f} and {mae_sklearn:.4f} for base")


if __name__ == '__main__':
    unittest.main()