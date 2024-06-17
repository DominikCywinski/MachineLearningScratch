import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def euclidean_distance(sample1, sample2):
    return np.sqrt(np.sum(np.square(sample1 - sample2)))


class KMeans:
    def __init__(self, n_clusters=3, max_iter=5, plot_steps=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.n_clusters)]
        self.centroids = []
        self.X = None
        self.num_samples = None
        self.num_features = None

    def predict(self, X):
        self.X = X
        self.num_samples, self.num_features = X.shape

        random_sample_idxs = np.random.choice(
            self.num_samples, self.n_clusters, replace=False
        )
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iter):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            prev_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(prev_centroids, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.num_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.n_clusters, self.num_features))

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, prev_centroids, centroids):
        distances = [
            euclidean_distance(prev_centroids[i], centroids[i])
            for i in range(self.n_clusters)
        ]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=5, s=100)

        plt.show()


if __name__ == "__main__":
    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=50
    )
    clusters = len(np.unique(y))
    k = KMeans(n_clusters=clusters, max_iter=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()
