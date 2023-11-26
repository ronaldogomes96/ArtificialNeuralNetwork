import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class RBF:

    def __init__(self, k_centers):
        self.k_centers = k_centers
        self.centers = None
        self.widths = None
        self.weights = None

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.k_centers, random_state=42)
        kmeans.fit(X)

        self.centers = kmeans.cluster_centers_

        self.widths = np.mean(euclidean_distances(self.centers, self.centers))
        distances = self.compute_rbf(X)
        self.weights = np.linalg.pinv(distances).dot(y)

    def compute_rbf(self, X):
        distances = euclidean_distances(X, self.centers)
        return np.exp(-(distances ** 2) / (2 * (self.widths ** 2)))

    def predict_classification(self, X):
        phi = self.compute_rbf(X)
        result = phi.dot(self.weights)
        return 0.5*(np.sign(result-0.5)+1)

    def predict_regression(self, X):
        phi = self.compute_rbf(X)
        return phi.dot(self.weights)
