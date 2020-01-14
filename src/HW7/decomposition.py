import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from KNN import KNN
from scipy.spatial.distance import pdist, squareform, cdist


class PCA:
    def __init__(self, X, n_components=25, kernel=None, gamma=1e-5):
        self.X = X
        self.n_samples = X.shape[1]
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.calculate_eigen()

    def calculate_eigen(self):
        self.mu = np.mean(self.X, axis=0)
        self.X = self.X - self.mu

        if self.kernel == 'rbf':
            sq_dists = pdist(self.X, 'sqeuclidean')
            mat_sq_dists = squareform(sq_dists)
            K = np.exp(-self.gamma * mat_sq_dists)
            N = K.shape[0]
            one_n = np.ones((N, N)) / N
            K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

            eig_values, eig_vectors = np.linalg.eigh(K)
            eig_values = np.flip(eig_values)
            eig_vectors = np.flip(eig_vectors, axis=1)
        elif self.kernel == 'linear':
            pass
        else:
            S = np.dot(self.X, self.X.T)
            print('S.shape', S.shape)
            eig_values, eig_vectors = np.linalg.eigh(S)
            eig_vectors = np.dot(self.X.T, eig_vectors)
            print('eig_vectors.shape:', eig_vectors.shape)
            eig_vectors = np.flip(eig_vectors, axis=1)
            eig_vectors = np.true_divide(eig_vectors, np.linalg.norm(eig_vectors, ord=2, axis=0).reshape(1, -1))
            eig_values = np.flip(eig_values)

        self.eig_vectors = eig_vectors[:, :self.n_components]
        self.eig_values = eig_values[:self.n_components]

        print(self.eig_values)
        plt.plot(self.eig_values)
        plt.show()

    def transform(self, X):
        if self.kernel == 'rbf':
            pair_dist = cdist(X.T, self.X.T, 'sqeuclidean')
            k = np.exp(-self.gamma * pair_dist)

            return k.dot(self.eig_vectors[:, :self.n_components] / self.eig_values[:self.n_components])
        else:
            return np.dot(X - self.mu, self.eig_vectors)

    def reconstruct(self, X):
        return np.dot(X, self.eig_vectors.T) + self.mu

    def set_n_components(self, n_components):
        self.n_components = n_components


class LDA:
    def __init__(self, X, y, n_components=25):
        self.X = X
        self.y = y
        self.n_classes = np.amax(y) + 1
        self.n_samples = X.shape[1]
        self.n_components = n_components
        self.calculate_eigen()

    def calculate_eigen(self):
        mean_vectors = []
        for c in range(self.n_classes):
            mean_vectors.append(np.mean(self.X[self.y == c], axis=0))
        self.mean_vectors = np.asarray(mean_vectors)

        print(self.mean_vectors.shape)
        d = self.X.shape[1]
        S_w = np.zeros((d, d))

        for label in range(self.n_classes):
            class_scatter = np.cov(self.X[self.y == label].T)
            S_w += class_scatter

        print('Scaled within-class scatter matrix: %sx%s' % (S_w.shape[0], S_w.shape[1]))

        # calculate between-class scatter matrix
        overall_mean = np.mean(self.X, axis=0).reshape(-1, 1)

        S_b = np.zeros((d, d))

        for i, mean_vector in enumerate(self.mean_vectors):
            n = self.X[self.y == i].shape[0]
            mean_vector = mean_vector.reshape(-1, 1)
            S_b += n * (mean_vector - overall_mean).dot((mean_vector - overall_mean).T)

        print('between-class Scatter Matrix:\n', S_b)

        eig_values, eig_vectors = np.linalg.eigh(np.linalg.inv(S_w).dot(S_b))
        self.eig_vectors = eig_vectors
        self.eig_values = eig_values

        eigen_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

        print('Eigendecomposition: \nEigenvalues in decreasing order:\n')
        self.w = np.hstack([eigen_pair[1].reshape(-1, 1).real for eigen_pair in eigen_pairs[:self.n_components]])
        print('Matrix W:\n', self.w)
        print('Matrix W:\n', self.w.shape)

    def transform(self, X):
        return X.dot(self.w)
