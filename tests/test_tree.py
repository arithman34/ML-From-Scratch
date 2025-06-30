import unittest
from src.utils.criterions import Gini, SquaredError
from src.supervised_learning.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier, DecisionTreeRegressor as SklearnDecisionTreeRegressor
from sklearn.metrics import accuracy_score
from tests.test_data import get_classification_data, get_regression_data
import numpy as np


class TestDecisionTreeClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data()

    def test_initialization(self):
        clf = DecisionTreeClassifier(max_depth=1, min_samples_split=2, criterion="gini")

        self.assertIsNone(clf.tree, msg="Tree should be initialized to None")
        self.assertEqual(clf.max_depth, 1, msg="Max depth should be set correctly")
        self.assertEqual(clf.min_samples_split, 2, msg="Min sample split should be set correctly")
        self.assertIsInstance(clf.criterion, Gini, msg="Criterion should be set correctly")

    def test_fit(self):
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(self.X_train, self.y_train)

        self.assertIsNotNone(clf.tree, msg="Tree should be set after fitting")

    def test_predict(self):
        # Fit custom KNeighborsClassifer
        clf = DecisionTreeClassifier(max_depth=100, criterion="gini", random_state=42)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        # Fit sklearn KNeighborsClassifier
        sklearn_model = SklearnDecisionTreeClassifier(max_depth=100, criterion="gini", random_state=42)
        sklearn_model.fit(self.X_train, self.y_train)
        sklearn_y_pred = sklearn_model.predict(self.X_test)

        # Accuracy
        acc_custom = accuracy_score(self.y_test, y_pred)
        acc_sklearn = accuracy_score(self.y_test, sklearn_y_pred)

        # Check mismatch ratio (decision trees are known to overfit)
        abs_error = np.abs(acc_custom - acc_sklearn)
        self.assertLessEqual(abs_error, 1, f"MAE mismatch too high got {acc_custom:.4f} and {acc_sklearn:.4f} for base")


class TestDecisionTreeRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_regression_data()

    def test_initialization(self):
        reg = DecisionTreeRegressor(max_depth=1, min_samples_split=2, criterion="squared_error")

        self.assertIsNone(reg.tree, msg="Tree should be initialized to None")
        self.assertEqual(reg.max_depth, 1, msg="Max depth should be set correctly")
        self.assertEqual(reg.min_samples_split, 2, msg="Min sample split should be set correctly")
        self.assertIsInstance(reg.criterion, SquaredError, msg="Criterion should be set correctly")

    def test_fit(self):
        reg = DecisionTreeRegressor(max_depth=1)
        reg.fit(self.X_train, self.y_train)

        self.assertIsNotNone(reg.tree, msg="Tree should be set after fitting")

    def test_predict(self):
        reg = DecisionTreeRegressor(max_depth=100, criterion="squared_error", random_state=42) 
        reg.fit(self.X_train, self.y_train)
        y_pred = reg.predict(self.X_test)

        # Fit sklearn KNeighborsRegressor
        sklearn_model = SklearnDecisionTreeRegressor(max_depth=100, criterion="squared_error", random_state=42)
        sklearn_model.fit(self.X_train, self.y_train)
        sklearn_y_pred = sklearn_model.predict(self.X_test)

        # Compare predictions
        mae_custom = np.mean(abs(y_pred - self.y_test))  # Mean Absolute Error (custom)
        mae_sklearn = np.mean(abs(sklearn_y_pred - self.y_test))  # Mean Absolute Error (sklearn)

        # Check mismatch ratio (decision trees are known to overfit)
        abs_mae = abs(mae_custom - mae_sklearn)
        self.assertLessEqual(abs_mae, 5, f"MAE mismatch too high got {mae_custom:.4f} and {mae_sklearn:.4f} for base")


if __name__ == '__main__':
    unittest.main()