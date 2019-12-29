import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolor
from sklearn.cluster import SpectralClustering as clustering
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import squareform, pdist
import kmeans


def load_image(file, size=None):
    im = Image.open(file)

    if size is not None:
        im.thumbnail(size)
        # im = im.crop((0, 0, 20, 20))
    width, height = im.size
    im = np.array(im, dtype=float).reshape((width * height, 3))

    return im


class SpectralClustering:
    def __init__(self, save_plot=None, n_clusters=2, normalized=True, max_iter=300, init='k-means++'):
        self.normalized = normalized
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.save_plot = save_plot

    def compute_rbf_kernel(self, X, gamma):
        return np.exp(-gamma * squareform(pdist(X, 'sqeuclidean')))

    def construct_similarity_matrix(self, X, n_rows, n_cols, gamma_c=None, gamma_s=None):
        coor = np.array([[i // n_cols, i % n_cols] for i in range(n_cols * n_rows)])

        if gamma_s is None:
            gamma_s = 1.0 / (100 * 100)

        if gamma_c is None:
            gamma_c = 1.0 / (255 * 255)

        color_similarity = self.compute_rbf_kernel(X, gamma_c)
        spatial_similarity = self.compute_rbf_kernel(coor, gamma_s)

        return color_similarity * spatial_similarity

    def ncut(self, w, d):
        degree_matrix_sq = np.diag(np.power(np.diag(d), -0.5))
        L_sym = np.eye(w.shape[0]) - degree_matrix_sq @ w @ degree_matrix_sq
        eigen_values, eigen_vectors = np.linalg.eig(L_sym)
        idx = np.argsort(eigen_values)[1: self.n_clusters + 1]
        U = eigen_vectors[:, idx].real.astype(np.float32)

        # normalized
        sum_over_row = (np.sum(np.power(U, 2), axis=1) ** 0.5).reshape(-1, 1)
        T = U.copy()
        for i in range(sum_over_row.shape[0]):
            if sum_over_row[i][0] == 0:
                sum_over_row[i][0] = 1
            T[i][0] /= sum_over_row[i][0]
            T[i][1] /= sum_over_row[i][0]

        return T

    def fit(self, X):
        n_rows = n_cols = int(np.sqrt(X.shape[0]))

        similarity_matrix = self.construct_similarity_matrix(X, n_rows, n_cols)
        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))

        if self.normalized:
            T = self.ncut(similarity_matrix, degree_matrix)
        else:
            T = self.rcut(similarity_matrix, degree_matrix)

        classifications = kmeans.KMeans(save_plots=self.save_plot, n_clusters=self.n_clusters, init=self.init).fit(T)
        colors = np.array(list(mcolor.TABLEAU_COLORS.keys()))
        print(T.shape)
        plt.scatter(T[:, 0], T[:, 1], c=colors[classifications])
        plt.show()

        return classifications

    def rcut(self, w, d):
        laplacian_matrix = d - w
        eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
        idx = np.argsort(eigen_values)[1: self.n_clusters + 1]
        U = eigen_vectors[:, idx].real.astype(np.float32)

        return U


def main():
    methods = ['k-means++', 'random']
    normalized = [True, False]
    for i in range(1, 3):
        img = load_image('./image%d.png' % i, size=(25, 25))
        for k in range(2, 3):
            for method in methods:
                for norm in normalized:
                    print('Image: {} Normalized: {} n_cluster: {} {}'.format(i, norm, k, method))
                    file = './img/image{}_{}_{}_{}'.format(i, 'ncut' if norm else 'rcut', k, method)
                    spectral_clustering = SpectralClustering(save_plot=None, n_clusters=k, normalized=norm, init=method)
                    classifications = spectral_clustering.fit(img).reshape(25, 25)

                    print('classification: ', classifications)
                    print('-------------------------------------------------------------')
    #
    # clusters = clustering(n_clusters=4).fit(img)
    #
    # print(clusters.labels_)


if __name__ == "__main__":
    main()
