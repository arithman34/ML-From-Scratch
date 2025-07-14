import numpy as np
from typing import Optional
from src.utils.criterions import Criteria, _initialize_criterion


class Leaf:  # Leaf node
    def __init__(self, value):
        self.value = value


class Node:  # Decision node
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion: Optional[Criteria] = None
        self.max_features = max_features

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        self.tree = self._fit(X, y, depth=0)

    def _fit(self, X, y, depth) -> Node:
        # Stopping conditions
        if self._should_stop(y, depth):
            return self._create_leaf(y)

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return self._create_leaf(y)

        # Make split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._fit(X[right_mask], y[right_mask], depth + 1)

        return Node(best_feature, best_threshold, left_subtree, right_subtree)

    def _should_stop(self, y, depth):
        return (len(set(y)) == 1 or depth == self.max_depth or len(y) < self.min_samples_split)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, int]:
        _, num_features = X.shape
        best_metric = float("inf")
        best_feature, best_threshold = None, None

        if self.max_features == "sqrt":
            num_selected_features = max(1, int(np.sqrt(num_features)))  # Prevent zero features
        elif self.max_features == "log2":
            num_selected_features = max(1, int(np.log2(num_features)))  # Prevent zero features
        elif isinstance(self.max_features, int):
            num_selected_features = self.max_features
        else:
            num_selected_features = num_features

        selected_features = np.random.choice(num_features, num_selected_features, replace=False)

        for feature in selected_features:
            feature_values = X[:, feature]
            thresholds = np.percentile(feature_values, np.linspace(0, 100, num=10))

            for threshold in thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_y = y[left_indices]
                right_y = y[right_indices]

                metric = self._split_metric(left_y, right_y)
                if metric < best_metric:
                    best_metric, best_feature, best_threshold = metric, feature, threshold

        return best_feature, best_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node):
        if isinstance(node, Leaf):
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)

    def _split_metric(self, left_y, right_y):
        left = self.criterion(left_y)
        right = self.criterion(right_y)
        total_samples = len(left_y) + len(right_y)
        return (len(left_y) / total_samples) * left + (len(right_y) / total_samples) * right

    def _create_leaf(self, y):
        raise NotImplementedError


class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=100, min_samples_split=2, criterion="gini", max_features=None, random_state=None):
        super().__init__(max_depth, min_samples_split, max_features, random_state)
        self.criterion = _initialize_criterion(criterion, is_classification=True)

    def _create_leaf(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        return Leaf(unique_classes[np.argmax(class_counts)])


class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=100, min_samples_split=2, criterion="squared_error", max_features=None, random_state=None):
        super().__init__(max_depth, min_samples_split, max_features, random_state)
        self.criterion = _initialize_criterion(criterion, is_classification=False)

    def _create_leaf(self, y):
        return Leaf(np.mean(y))
