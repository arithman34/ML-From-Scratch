import unittest
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from tests.test_data import get_classification_data
from src.unsupervised_learning.pca import PCA


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, _, _, _ = get_classification_data()

    def test_pca_transform(self):
        pca_sklearn = SklearnPCA(n_components=2)
        pca_custom = PCA(n_components=2)

        # Fit and transform
        X_sklearn_pca = pca_sklearn.fit_transform(self.X)
        X_custom_pca = pca_custom.fit_transform(self.X)

        # https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative
        np.testing.assert_array_almost_equal(abs(X_sklearn_pca), abs(X_custom_pca), decimal=7, err_msg="PCA fit and transform mismatch")

    def test_pca_components(self):
        pca_sklearn = SklearnPCA(n_components=2)
        pca_custom = PCA(n_components=2)

        # Fit PCA models
        pca_sklearn.fit(self.X)
        pca_custom.fit(self.X)

        # https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative
        np.testing.assert_array_almost_equal(abs(pca_sklearn.components_), abs(pca_custom.components_), decimal=7, err_msg="PCA component mismatch")

    def test_pca_explained_variance(self):
        pca_sklearn = SklearnPCA(n_components=2)
        pca_custom = PCA(n_components=2)
        
        # Fit PCA models
        pca_sklearn.fit(self.X)
        pca_custom.fit(self.X)

        np.testing.assert_array_almost_equal(pca_sklearn.explained_variance_ratio_, pca_custom.explained_variance_ratio_, decimal=5)


if __name__ == '__main__':
    unittest.main()
