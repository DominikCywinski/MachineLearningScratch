import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from get_accuracy import get_accuracy


class SupportVectorMachine:
    def __init__(
        self, learning_rate=0.001, lambda_param=0.01, num_iterations=1000
    ):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = (
                    y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                )
                if condition:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                        - np.dot(y_[idx], x_i)
                    )
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, classifier.weights, classifier.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, classifier.weights, classifier.bias, 0)

    x1_1_m = get_hyperplane_value(x0_1, classifier.weights, classifier.bias, -1)
    x1_2_m = get_hyperplane_value(x0_2, classifier.weights, classifier.bias, -1)

    x1_1_p = get_hyperplane_value(x0_1, classifier.weights, classifier.bias, 1)
    x1_2_p = get_hyperplane_value(x0_2, classifier.weights, classifier.bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()


if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, random_state=50
    )
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    classifier = SupportVectorMachine()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = get_accuracy(predictions, y_test)
    print("Accuracy on test set: {:.2f}".format(accuracy))
    visualize_svm()
