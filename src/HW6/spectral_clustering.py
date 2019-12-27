import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import SpectralClustering as clustering
from sklearn.metrics.pairwise import rbf_kernel
import kmeans


def load_image(file, size=None):
    im = Image.open(file)

    if size is not None:
        # im.thumbnail(size)
        im = im.crop((0, 0, 20, 20))
    width, height = im.size
    im = np.array(im, dtype=float).reshape((width * height, 3))
    print(width)
    print(height)
    print(im)

    return im


class SpectralClustering:
    def __init__(self, n_clusters=2, normalized=True, max_iter=300):
        self.normalized = normalized
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def compute_rbf_kernel(self, X, X_prime, gamma):
        X_norm = np.matmul(X, X_prime.T)
        K = X_norm * (-gamma)
        np.exp(K, K)

        return K

    def construct_similarity_matrix(self, X, n_rows, n_cols, gamma_c=None, gamma_s=None):
        coor = np.array([[i // n_cols, i % n_cols] for i in range(n_cols * n_rows)])

        if gamma_s is None:
            gamma_s = 1 / 2

        if gamma_c is None:
            gamma_c = 1 / 3

        ss = rbf_kernel(coor, gamma=gamma_s)
        cs = rbf_kernel(X, gamma=gamma_c)
        print('sklearn spatial similarity:\n', ss)
        print('sklearn color similarity:\n', cs)

        color_similarity = self.compute_rbf_kernel(X, X, gamma_c)
        spatial_similarity = self.compute_rbf_kernel(coor, coor, gamma_s)
        print('our color_similarity:\n', color_similarity)
        print('our spatial_similarity:\n', spatial_similarity)

        return np.multiply(ss, cs)

    def fit(self, X):
        n_rows = n_cols = int(np.sqrt(X.shape[0]))
        similarity_matrix = self.construct_similarity_matrix(X, n_rows, n_cols)

        print('similarity_matrix.shape: ', similarity_matrix.shape)

        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))

        laplacian_matrix = similarity_matrix - degree_matrix
        eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
        idx = np.argsort(eigen_values)[1: self.n_clusters + 1]
        U = eigen_vectors[:, idx].real.astype(np.float32)

        print(U.shape)

        classifications = kmeans.KMeans(n_clusters=self.n_clusters).fit(U)

        return classifications


def main():
    img = load_image('./image1.png', size=(25, 25))
    # spectral_clustering = SpectralClustering(n_clusters=4)
    # classifications = spectral_clustering.fit(img)
    #
    # print('classification: ', classifications)

    clusters = clustering(n_clusters=4).fit(img)

    print(clusters.labels_)


if __name__ == "__main__":
    main()
