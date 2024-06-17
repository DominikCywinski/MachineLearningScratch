import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from get_accuracy import get_accuracy


def unit_step_function(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_function = unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted


def model_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.fit(X_train, y_train)
    predictions = perceptron.predict(X_test)

    accuracy = get_accuracy(predictions, y_test)
    print("Accuracy on test set: {:.2f}".format(accuracy))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (
        -perceptron.weights[0] * x0_1 - perceptron.bias
    ) / perceptron.weights[1]
    x1_2 = (
        -perceptron.weights[0] * x0_2 - perceptron.bias
    ) / perceptron.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ax.set_ylim([np.amin(X_train[:, 1]) - 3, np.amax(X_train[:, 1]) + 3])

    plt.show()


if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=100, n_features=2, centers=2, random_state=0
    )
    model_test(X, y)
