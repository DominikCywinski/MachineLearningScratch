import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from get_accuracy import get_accuracy


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = []
        self.bias = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            predicts = 1 / (
                1 + np.exp(-1 * (np.dot(X, self.weights) + self.bias))
            )  # sigmoid function
            dw = (1 / num_samples) * np.dot(X.T, (predicts - y))
            db = (1 / num_samples) * np.sum(predicts - y)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_predicts = 1 / (
            1 + np.exp(-1 * (np.dot(X, self.weights) + self.bias))
        )  # sigmoid
        return [0 if y <= 0.5 else 1 for y in y_predicts]


def model_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_test)

    accuracy = get_accuracy(y_predictions, y_test)
    print("Accuracy: {:.2f}".format(accuracy))


if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    model_test(X, y)
