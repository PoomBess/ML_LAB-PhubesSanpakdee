import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class TFKNNClassifier:

    def __init__(self, k=5):
        self.k = k  # จำนวนเพื่อนบ้านที่ใช้
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        # Train KNN
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """predict class of new data and return array of class labels"""
        return self.model.predict(X)

    def score(self, X, y):
        """calculate accuracy = proportion of correct predictions"""
        return float(np.mean(self.predict(X) == y))