import numpy as np
import unittest
from src.supervised_learning.naive_bayes import MultinomialNB, GaussianNB
from src.data.data_generator import get_classification_data
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB, GaussianNB as SklearnGaussianNB


class TestNaiveBayes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data(num_samples=10000, num_classes=2, normalize=True)

    def test_multinomial_naive_bayes(self):
        np.random.seed(42)
        clf_custom = MultinomialNB(alpha=1.0)
        clf_custom.fit(self.X_train, self.y_train)
        y_pred_custom = clf_custom.predict(self.X_test)

        clf_sklearn = SklearnMultinomialNB(alpha=1.0)
        clf_sklearn.fit(self.X_train, self.y_train)
        y_pred_sklearn = clf_sklearn.predict(self.X_test)

        acc_custom = accuracy_score(self.y_test, y_pred_custom)
        acc_sklearn = accuracy_score(self.y_test, y_pred_sklearn)
        abs_acc = abs(acc_custom - acc_sklearn)

        self.assertLessEqual(abs_acc, 0.05, f"Accuracy mismatch too high: {acc_custom:.4f} vs {acc_sklearn:.4f}")


class TestGaussianNaiveBayes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = get_classification_data(num_samples=10000, num_classes=2, normalize=False)

    def test_gaussian_naive_bayes(self):
        clf_custom = GaussianNB(var_smoothing=1e-9)
        clf_custom.fit(self.X_train, self.y_train)
        y_pred_custom = clf_custom.predict(self.X_test)

        clf_sklearn = SklearnGaussianNB(var_smoothing=1e-9)
        clf_sklearn.fit(self.X_train, self.y_train)
        y_pred_sklearn = clf_sklearn.predict(self.X_test)

        acc_custom = accuracy_score(self.y_test, y_pred_custom)
        acc_sklearn = accuracy_score(self.y_test, y_pred_sklearn)
        abs_acc = abs(acc_custom - acc_sklearn)

        self.assertLessEqual(abs_acc, 0.05, f"Accuracy mismatch too high: {acc_custom:.4f} vs {acc_sklearn:.4f}")
