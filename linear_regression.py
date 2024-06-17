import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt


def mse(tests, predictions):
    return np.mean((tests - predictions) ** 2)


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.learning_rate = lr
        self.num_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


def model_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], y, color="r", marker="x")
    plt.title("Input Data", fontsize=20, fontweight="bold")
    plt.show()

    linear_regression = LinearRegression(lr=0.01)
    linear_regression.fit(X_train, y_train)
    predictions = linear_regression.predict(X_test)

    print("Mean squared error: %.2f" % mse(y_test, predictions))

    y_pred_line = linear_regression.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 8))
    m_train = plt.scatter(X_train, y_train, color=cmap(0.7))
    m_test = plt.scatter(X_test, y_test, color=cmap(0.2))
    plt.plot(X, y_pred_line, color="black", linewidth=1, label="Prediction")
    plt.title("Linear Regression", fontsize=20, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    X, y = datasets.make_regression(
        n_samples=200, n_features=1, noise=30, random_state=0
    )
    model_test(X, y)
