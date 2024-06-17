import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


class KNN:
    def __init__(self, k=3):
        self.X_train = None
        self.y_train = None
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_nearest_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        return Counter(k_nearest_labels).most_common()[0][0]


def model_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print("Accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    iris_data = datasets.load_iris()
    X, y = iris_data.data, iris_data.target
    model_test(X, y)
