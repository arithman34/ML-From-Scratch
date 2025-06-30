from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split


def get_classification_data():
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def get_regression_data():
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        noise=10.0,
        random_state=42
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_clustering_data():
    # Generate synthetic clustering data
    X, y = make_blobs(
        n_samples=1000,
        n_features=2,
        centers=3,
        random_state=42
    )
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)