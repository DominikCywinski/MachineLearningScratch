import numpy as np
from decision_tree import DecisionTree
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from get_accuracy import get_accuracy


class RandomForest:
    def __init__(
        self, num_trees=5, max_depth=10, min_samples_split=2, num_features=None
    ):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                num_features=self.num_features,
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X, y)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        num_samples = X.shape[0]
        idxs = np.random.choice(num_samples, num_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predicts = np.swapaxes(predictions, 0, 1)
        return np.array(
            [self._most_common_label(pred) for pred in tree_predicts]
        )


def model_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    clf = RandomForest()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = get_accuracy(y_test, predictions)
    print("Accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    model_test(X, y)
