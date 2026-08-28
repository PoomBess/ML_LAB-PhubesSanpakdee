## KNN with mean function (for clustering)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNClusterAssigner:

    def __init__(self, k=5):
        self.k = k

    # -----------------------------------------------------------------
    def fit(self, X, cluster_labels):

        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(X, cluster_labels)
        return self

    # -----------------------------------------------------------------
    def predict(self, X_new):

        return self.model.predict(X_new).astype("int32")