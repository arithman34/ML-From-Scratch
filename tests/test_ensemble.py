import unittest
from src.supervised_learning.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from tests.test_data import get_classification_data, get_regression_data
import numpy as np


class TestRandomForestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data()

    def test_initialization(self):
        clf = RandomForestClassifier(n_estimators=1, max_depth=1, min_samples_split=2, criterion="gini")

        self.assertEqual(len(clf.trees), 0, msg="Trees should be empty")
        self.assertEqual(clf.n_estimators, 1, msg="Number of estimators should be set correctly")
        self.assertEqual(clf.max_depth, 1, msg="Max depth should be set correctly")
        self.assertEqual(clf.min_samples_split, 2, msg="Min sample split should be set correctly")

    def test_fit(self):
        clf = RandomForestClassifier(n_estimators=10, max_depth=100)
        clf.fit(self.X_train, self.y_train)

        self.assertEqual(len(clf.trees), 10, msg="Trees should equal to number of estimators")

    def test_predict(self):
        clf = RandomForestClassifier(n_estimators=100, max_depth=100, criterion="gini", max_features="sqrt", random_state=42)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        acc_custom = accuracy_score(self.y_test, y_pred)

        self.assertGreaterEqual(acc_custom, 0.9, f"Accuracy should be higher, got {acc_custom:.2f}")


class TestRandomForestRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data()

    def test_initialization(self):
        reg = RandomForestRegressor(n_estimators=1, max_depth=1, min_samples_split=2, criterion="squared_error")

        self.assertEqual(len(reg.trees), 0, msg="Trees should be empty")
        self.assertEqual(reg.n_estimators, 1, msg="Number of estimators should be set correctly")
        self.assertEqual(reg.max_depth, 1, msg="Max depth should be set correctly")
        self.assertEqual(reg.min_samples_split, 2, msg="Min sample split should be set correctly")

    def test_fit(self):
        reg = RandomForestRegressor(n_estimators=10, max_depth=1)
        reg.fit(self.X_train, self.y_train)

        self.assertEqual(len(reg.trees), 10, msg="Trees should equal to number of estimators")

    def test_predict(self):
        reg = RandomForestRegressor(n_estimators=100, max_depth=100, criterion="squared_error", max_features=1, random_state=42)
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)

        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error (custom)

        self.assertLessEqual(mae_custom, 60, f"Mean Absolute Error should be lower, got {mae_custom:.2f}")


if __name__ == '__main__':
    unittest.main()