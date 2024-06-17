import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from get_accuracy import get_accuracy


class NaiveBayes:

    def __init__(self):
        self._mean = None
        self._var = None
        self._priors = None
        self._classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = np.mean(X_c, axis=0)
            self._var[idx, :] = np.var(X_c, axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_predictions = [self._predict(x) for x in X]
        return np.array(y_predictions)

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(
                np.log(self._probability_density_function(idx, x))
            )
            posterior = posterior + prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _probability_density_function(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator


def model_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = get_accuracy(y_test, y_pred)
    print("Accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    model_test(X, y)
