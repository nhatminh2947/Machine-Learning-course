import numpy as np
from scipy.spatial.distance import cdist, sqeuclidean
import matplotlib.pyplot as plt

def k_mean_plus_plus_init(X, n_clusters):
    n_samples, n_features = X.shape
    centers = []
    center_id = np.random.randint(n_samples)
    centers.append(X[center_id])

    for i in range(1, n_clusters):
        distance = cdist(X, centers)
        print('distance: ', distance)
        center_id = np.argmax(distance[np.arange(n_samples), np.argmin(distance, axis=1)])
        centers = np.vstack((centers, X[center_id]))

    return centers


def init_centroids(X, n_clusters, init='k-means++'):
    n_samples = X.shape[0]

    if init == 'k-means++':
        centers = k_mean_plus_plus_init(X, n_clusters)
    elif init == 'random':
        seeds = np.random.permutation(n_samples)[:n_clusters]
        centers = X[seeds]

    return centers


class KMeans:
    def __init__(self, init='k-means++', n_clusters=2, max_iter=300):
        self.max_iter = max_iter
        self.init = init
        self.n_clusters = n_clusters

    def fit(self, X):
        global classifications
        centroids = init_centroids(X, self.n_clusters, self.init)

        for i in range(self.max_iter):
            print('iter: ', i)
            old_centroids = centroids.copy()
            distance = cdist(X, centroids)
            classifications = np.argmin(distance, axis=1)

            for k in range(self.n_clusters):
                ids = np.where(classifications == k)[0]
                centroids[k] = np.mean(X[ids, :], axis=0)

            error = self.calculate_error(old_centroids, centroids)

            if error < 1e-6:
                break

        print('centers', centroids)
        return classifications

    def calculate_error(self, old_centroids, new_centroids):
        error = 0
        for i in range(self.n_clusters):
            error += sqeuclidean(old_centroids[i], new_centroids[i])

        return error


a = np.random.random(size=(100, 2))
print('a: ', a)
predicted = KMeans(n_clusters=3).fit(a)

print(predicted)
c1 = a[np.where(predicted == 0)]
c2 = a[np.where(predicted == 1)]
c3 = a[np.where(predicted == 2)]
print(c1)
print(c2)

plt.scatter(c1[:, 0], c1[:, 1])
plt.scatter(c2[:, 0], c2[:, 1])
plt.scatter(c3[:, 0], c3[:, 1])

plt.show()