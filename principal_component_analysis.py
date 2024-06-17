import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class PrincipalComponentAnalysis:
    def __init__(self, num_components):
        self.num_components = num_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[: self.num_components]

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target
    pca = PrincipalComponentAnalysis(3)
    pca.fit(X)
    X_projected = pca.transform(X)
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y)
    plt.show()
