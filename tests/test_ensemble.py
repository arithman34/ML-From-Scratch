import unittest
from src.supervised_learning.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier, RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.metrics import accuracy_score
from src.data.data_generator import get_classification_data, get_regression_data
import numpy as np


class TestRandomForestClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data(num_samples=10000, num_classes=10)

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
        clf = RandomForestClassifier(n_estimators=10, max_depth=100, criterion="gini", max_features="sqrt", random_state=42)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        sklearn_clf = SklearnRandomForestClassifier(n_estimators=10, max_depth=100, criterion="gini", max_features="sqrt", random_state=42)
        sklearn_clf.fit(self.X_train, self.y_train)
        sklearn_y_pred = sklearn_clf.predict(self.X_test)

        acc_custom = accuracy_score(self.y_test, y_pred)
        acc_sklearn = accuracy_score(self.y_test, sklearn_y_pred)
        abs_error = np.abs(acc_custom - acc_sklearn)

        self.assertLessEqual(abs_error, 0.05, f"MAE mismatch too high got {acc_custom:.4f} and {acc_sklearn:.4f} for base")


class TestRandomForestRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data(num_samples=1000)

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

        sklearn_reg = SklearnRandomForestRegressor(n_estimators=100, max_depth=100, criterion="squared_error", max_features=1, random_state=42)
        sklearn_reg.fit(self.X_train, self.y_train)
        sklearn_y_pred = sklearn_reg.predict(self.X_test)

        mae_custom = np.mean(abs(y_pred - self.y_test))
        mae_sklearn = np.mean(abs(sklearn_y_pred - self.y_test))
        abs_error = np.abs(mae_custom - mae_sklearn)

        self.assertLessEqual(abs_error, 10.0, f"MAE mismatch too high got {mae_custom:.4f} and {mae_sklearn:.4f} for base")


if __name__ == '__main__':
    unittest.main()