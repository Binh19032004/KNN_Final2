import numpy as np
from collections import Counter


class MyKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = distances.argsort()[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])



