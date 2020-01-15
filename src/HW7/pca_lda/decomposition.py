import numpy as np
from scipy.spatial.distance import cdist


def poly_kernel(X, X_hat=None, degree=3):
    if X_hat is None:
        X_hat = X

    return (X.dot(X_hat.T) + 1) ** degree


def rbf_kernel(X, X_hat=None, gamma=1e-5):
    if X_hat is None:
        X_hat = X

    sq_dists = cdist(X, X_hat, 'sqeuclidean')

    return np.exp(-gamma * sq_dists)


class PCA:
    def __init__(self, X, n_components=25, kernel=None, gamma=None, degree=3):
        if gamma is None:
            self.gamma = 1.0 / X.shape[1]
        else:
            self.gamma = gamma

        self.X = X
        self.n_samples = X.shape[0]
        self.n_components = n_components
        self.degree = degree
        self.kernel = kernel
        self.calculate_eigen()

    def calculate_eigen(self):
        self.mu = np.mean(self.X, axis=0)
        self.X = self.X - self.mu

        if self.kernel is not None:
            if self.kernel == 'rbf':
                K = rbf_kernel(X=self.X, gamma=self.gamma)
            elif self.kernel == 'poly':
                K = poly_kernel(X=self.X, degree=self.degree)

            N = K.shape[0]
            one_n = np.ones((N, N)) / N
            K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

            eig_values, eig_vectors = np.linalg.eigh(K)
            eig_values = np.flip(eig_values)
            eig_vectors = np.flip(eig_vectors, axis=1)
            self.eig_vectors = eig_vectors[:, :self.n_components].copy()
            self.eig_values = eig_values[:self.n_components].copy()
            self.eig_vectors = self.eig_vectors / np.sqrt(self.eig_values)
        else:
            S = np.dot(self.X, self.X.T)
            eig_values, eig_vectors = np.linalg.eigh(S)
            eig_values = np.flip(eig_values)
            eig_vectors = np.dot(self.X.T, eig_vectors)
            eig_vectors = np.flip(eig_vectors, axis=1)
            eig_vectors = np.true_divide(eig_vectors, np.linalg.norm(eig_vectors, ord=2, axis=0).reshape(1, -1))
            self.eig_vectors = eig_vectors[:, :self.n_components].copy()
            self.eig_values = eig_values[:self.n_components].copy()

    def transform(self, X):
        if self.kernel == 'rbf':
            K = rbf_kernel(X, self.X, self.gamma)
            return K.dot(self.eig_vectors)
        elif self.kernel == 'poly':
            K = poly_kernel(X, self.X, self.degree)
            return K.dot(self.eig_vectors)
        else:
            X = X - self.mu
            return np.dot(X, self.eig_vectors)

    def reconstruct(self, X):
        return np.dot(X, self.eig_vectors.T) + self.mu

    def set_n_components(self, n_components):
        self.n_components = n_components


class LDA:
    def __init__(self, X, y, kernel=None, gamma=1e-9, degree=3, n_components=25):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.n_classes = np.amax(y) + 1
        self.n_samples = X.shape[0]
        self.n_components = n_components
        self.calculate_eigen()

    def calculate_eigen(self):
        if self.kernel is None:
            mean_vectors = []
            for c in range(self.n_classes):
                mean_vectors.append(np.mean(self.X[self.y == c], axis=0))
            self.mean_vectors = np.asarray(mean_vectors)

            d = self.X.shape[1]
            S_w = np.zeros((d, d))

            for label in range(self.n_classes):
                class_scatter = np.cov(self.X[self.y == label].T)
                S_w += class_scatter
            overall_mean = np.mean(self.X, axis=0).reshape(-1, 1)
            S_b = np.zeros((d, d))
            for i, mean_vector in enumerate(self.mean_vectors):
                n = self.X[self.y == i].shape[0]
                mean_vector = mean_vector.reshape(-1, 1)
                S_b += n * (mean_vector - overall_mean).dot((mean_vector - overall_mean).T)

            eig_values, eig_vectors = np.linalg.eigh(np.linalg.pinv(S_w).dot(S_b))
            self.eig_vectors = eig_vectors
            self.eig_values = eig_values

            eigen_pairs = [(eig_values[i], eig_vectors[:, i]) for i in range(len(eig_values))]
            eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

            self.w = np.hstack([eigen_pair[1].reshape(-1, 1).real for eigen_pair in eigen_pairs[:self.n_components]])
        else:
            if self.kernel == 'rbf':
                K = rbf_kernel(X=self.X, gamma=self.gamma)
            elif self.kernel == 'poly':
                K = poly_kernel(X=self.X, degree=self.degree)

            overall_mean = np.mean(K, axis=0).reshape(-1, 1)

            mean_vectors = []
            for c in range(self.n_classes):
                mean_vectors.append(np.mean(K[self.y == c], axis=0))
            self.mean_vectors = np.asarray(mean_vectors)

            S_b = np.zeros([self.n_samples, self.n_samples])
            for i, mean_vector in enumerate(self.mean_vectors):
                n = self.X[self.y == i].shape[0]
                mean_vector = mean_vector.reshape(-1, 1)
                S_b += n * (mean_vector - overall_mean).dot((mean_vector - overall_mean).T)

            S_w = np.zeros([self.n_samples, self.n_samples])
            I_minus_one = np.identity(9) - (9 * np.ones((9, 9)))
            for i in range(self.n_classes):
                Ki = K[self.y == i].copy()
                S_w += Ki.T.dot(I_minus_one.dot(Ki))

            eig_values, eig_vectors = np.linalg.eigh(np.linalg.pinv(S_w).dot(S_b))
            self.eig_vectors = eig_vectors
            self.eig_values = eig_values

            eigen_pairs = [(eig_values[i], eig_vectors[:, i]) for i in range(len(eig_values))]
            eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

            self.w = np.hstack([eigen_pair[1].reshape(-1, 1).real for eigen_pair in eigen_pairs[:self.n_components]])

    def transform(self, X):
        if self.kernel == 'rbf':
            K = rbf_kernel(X, self.X, self.gamma)
            return K.dot(self.eig_vectors)
        elif self.kernel == 'poly':
            K = poly_kernel(X, self.X, self.degree)
            return K.dot(self.eig_vectors)
        return X.dot(self.w)

    def reconstruct(self, X):
        return np.dot(X, self.w.T)
