import numpy as np


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y
        self.n_classes = np.amax(y) + 1

    def predict(self, X, k_neighbors=-1):
        if k_neighbors == -1:
            k_neighbors = self.k

        predictions = []

        for x_test in X:
            distances = np.sum(np.square(self.x_train - x_test), axis=1)
            votes = np.zeros(self.n_classes, dtype=np.int)

            for neighbor_id in np.argsort(distances)[:k_neighbors]:
                neighbor_label = self.y_train[neighbor_id]
                votes[neighbor_label] += 1

            predictions.append(np.argmax(votes))

        return np.asarray(predictions)
