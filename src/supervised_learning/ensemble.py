import numpy as np
from src.supervised_learning.tree import DecisionTree, DecisionTreeClassifier, DecisionTreeRegressor


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.max_features = max_features
        
        if random_state is not None:  # Cannot call inside Decision Tree as all tree's will be identical
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        self.trees = [self._create_tree() for _ in range(self.n_estimators)]
        for tree in self.trees:
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
    
    def _create_tree(self) -> DecisionTree:
        raise NotImplementedError
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return self._aggregate_predictions(predictions)
    
    def _aggregate_predictions(self, predictions):
        raise NotImplementedError


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2, criterion="gini", max_features="sqrt", random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split, max_features, random_state)
        self.criterion = criterion
    
    def _create_tree(self) -> DecisionTree:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features
        )
    
    def _aggregate_predictions(self, predictions):
        # Majority voting
        return np.array([np.argmax(np.bincount(predictions[:, i])) for i in range(predictions.shape[1])])


class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=100, max_depth=100, min_samples_split=2, criterion="squared_error", max_features=1, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split, max_features, random_state)
        self.criterion = criterion
    
    def _create_tree(self) -> DecisionTree:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features
        )
    
    def _aggregate_predictions(self, predictions):
        # Averaging for regression
        return np.mean(predictions, axis=0)
