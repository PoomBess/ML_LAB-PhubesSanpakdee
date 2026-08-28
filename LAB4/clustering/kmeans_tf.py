## simple K-Means Clustering with NumPy/Scikit-Learn (Fallback version)

import numpy as np
from sklearn.cluster import KMeans


class TFKMeans:
    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=seed,
            n_init="auto",
        )

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.centroids_ = self.model.cluster_centers_
        self.n_iter_ = self.model.n_iter_
        self.inertia_ = float(self.model.inertia_)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_