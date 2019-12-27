import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import kmeans
import logging

logging.basicConfig(level=logging.DEBUG)


def load_image(file, size=None):
    im = Image.open(file)
    im.thumbnail(size)

    im = np.array(im)
    logging.info(im.shape)

    return im


class SpectralClustering:
    def __init__(self, n_clusters=2, normalized=True, max_iter=300, gamma_c=0.01, gamma_s=0.01):
        self.normalized = normalized
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s

    def compute_rbf_kernel(self, X, X_prime, gamma):
        X_norm = np.matmul(X, X_prime.T)
        K = X_norm * (-gamma)
        np.exp(K, K)

        return K

    def construct_similarity_matrix(self, X, n_rows, n_cols):
        coor = np.array([[i // n_cols, i % n_cols] for i in range(n_cols * n_rows)])

        color_similarity = self.compute_rbf_kernel(X, X, self.gamma_c)
        spatial_similarity = self.compute_rbf_kernel(coor, coor, self.gamma_s)

        return np.multiply(color_similarity, spatial_similarity)

    def fit(self, X):
        n_rows = n_cols = np.sqrt(X.shape[0])
        similarity_matrix = self.construct_similarity_matrix(X, n_rows, n_cols)
        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))

        laplacian_matrix = similarity_matrix - degree_matrix
        eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
        idx = np.argsort(eigen_values)[1: self.n_clusters + 1]
        U = eigen_vectors[:, idx].real.astype(np.float32)

        classifications = kmeans.KMeans(n_clusters=self.n_clusters).fit(U)

        return classifications


def main():
    img = load_image('./image1.png', size=(10, 10))
    logging.debug(img)
    spectral_clustering = SpectralClustering(n_clusters=2)
    classifications = spectral_clustering.fit(img)

    print('classification: ', classifications)


if __name__ == "__main__":
    main()
