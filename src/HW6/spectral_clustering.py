import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging

logging.basicConfig(level=logging.DEBUG)


def load_image(file, size=None):
    im = Image.open(file)
    im.thumbnail(size)

    im = np.array(im)
    logging.info(im.shape)

    return im


class SpectralClustering:
    def __init__(self, n_clusters=2, normalized=True, max_iter=1000):
        self.normalized = normalized
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def compute_rbf_kernel(self, X, X_prime, gamma):
        X_norm = np.matmul(X, X_prime.T)
        K = X_norm * (-gamma)
        np.exp(K, K)

        return K

    def gram_matrix(self, X):
        return X.dot(X.T)

    def custom_kernel(self, X, gamma_s=None, gamma_c=None):
        row, col = X.shape[0], X.shape[1]

        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])

        logging.info(X[0])

        if gamma_c is None:
            gamma_c = 1.0 / X.shape[1]

        if gamma_s is None:
            gamma_s = 1.0 / X.shape[1]

        rbf_c = self.compute_rbf_kernel(X, X, gamma_c)

        x = np.array([np.full(col, i) for i in range(row)]).reshape(row * col, 1)
        y = np.array([np.arange(col) for i in range(row)]).reshape(row * col, 1)

        xy = np.hstack([x, y])
        rbf_s = self.compute_rbf_kernel(xy, xy, gamma_s)

        logging.info(rbf_c[0][0])
        logging.info(rbf_s)

        return np.multiply(rbf_s, rbf_c)

    def ratio_cut(self, gram_matrix):
        degree = np.diag(np.sum(gram_matrix, axis=1))
        L = degree - gram_matrix
        logging.debug('L shape: {}'.format(L.shape))
        eigen_values, eigen_vectors = np.linalg.eig(L)
        idx = np.argsort(eigen_values)[1: self.n_clusters + 1]
        U = eigen_vectors[:, idx].real.astype(np.float32)

        return U

    def clustering(self, data):
        mu = np.random.randn(self.n_clusters, 2)
        classification = np.random.random_integers(low=1, high=self.n_clusters, size=data.shape[0])

        iterate = 0
        while iterate < self.max_iter:


            iterate += 1

    def fit(self, data):
        logging.debug(data.shape)

        weights = self.custom_kernel(data)

        U = self.ratio_cut(weights)
        logging.debug('U = {}'.format(U))

        logging.info(weights.shape)

        return 0


def main():
    img = load_image('./image1.png', size=(10, 10))
    logging.debug(img)
    spectral_clustering = SpectralClustering(n_clusters=2)
    spectral_clustering.fit(img)


if __name__ == "__main__":
    main()