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
        # center_id = np.argmax(distance[np.arange(n_samples), np.argmin(distance, axis=1)])
        distance = distance[np.arange(n_samples), np.argmin(distance, axis=1)]
        distance = distance / distance.sum()

        center_id = np.random.choice(n_samples, 1, p=distance)

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
    def __init__(self, save_plots, init='k-means++', n_clusters=2, max_iter=300):
        self.max_iter = max_iter
        self.init = init
        self.n_clusters = n_clusters
        self.save_plots = save_plots

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
            print('error: ', error)

            plt.matshow(classifications.reshape(25, 25))
            plt.savefig('{}_{}.png'.format(self.save_plots, i))

            if error < 1e-6:
                break

        return classifications

    def calculate_error(self, old_centroids, new_centroids):
        error = 0
        for i in range(self.n_clusters):
            error += sqeuclidean(old_centroids[i], new_centroids[i])

        return error
