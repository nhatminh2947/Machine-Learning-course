from matplotlib import cm
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import KNeighborsClassifier
import glob
import PIL.Image as Image
import numpy as np

rows = 195
cols = 231


def load_data(path):
    print('Loading... %s' % path)
    X = []
    y = []

    for id in range(0, 15):
        filelist = glob.glob(path + 'subject' + str(id + 1).zfill(2) + "*")
        for fname in filelist:
            img = Image.open(fname)
            print(img.size)
            img = np.array(img.resize((rows, cols), Image.ANTIALIAS))

            X.append(img)
            y.append(id)
    print('There are %d samples' % len(y))
    return np.asarray(X), np.asarray(y)


def fisherfaces(X, y, num_components=0):
    y = np.asarray(y)
    print(X.shape)
    [n, d] = X.shape
    c = len(np.unique(y))
    [eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, (n - c))
    [eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca, X, mu_pca), y, num_components)
    eigenvectors = np.dot(eigenvectors_pca, eigenvectors_lda)
    return [eigenvalues_lda, eigenvectors, mu_pca]


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


def pca(X, y, num_components=0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    # or simply perform an economy size decomposition
    # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


def lda(X, y, num_components=0):
    y = np.asarray(y)
    [n, d] = X.shape
    c = np.unique(y)
    if (num_components <= 0) or (num_components > (len(c) - 1)):
        num_components = (len(c) - 1)
    meanTotal = X.mean(axis=0)
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    for i in c:
        Xi = X[np.where(y == i)[0], :]
        meanClass = Xi.mean(axis=0)
        Sw = Sw + np.dot((Xi - meanClass).T, (Xi - meanClass))
        Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
    eigenvectors = np.array(eigenvectors[0:, 0:num_components].real, dtype=np.float32, copy=True)
    return [eigenvalues, eigenvectors]


def create_font(fontname='Tahoma', fontsize=10):
    return {'fontname': fontname, 'fontsize': fontsize}


import os, sys


def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            print(subject_path)
            for filename in os.listdir(subject_path):
                # try:
                im = Image.open(os.path.join(subject_path, filename))
                im = im.convert("L")
                # resize to given size (if given)
                if sz is not None:
                    im = im.resize(sz, Image.ANTIALIAS)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
                # except IOError:
                #     print("I/O error({0}): {1}".format(errno, strerror))
                # except:
                #     print
                #     "Unexpected error:", sys.exc_info()[0]
                #     raise
            c = c + 1
    return [X, y]


def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True,
            filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma', 10))
        else:
            plt.title("%s #%d" % (sptitle, (i + 1)), create_font('Tahoma', 10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


from KNN import KNN

if __name__ == '__main__':

    # read images
    [X, y] = load_data('./Yale_Face_Database/Training/')
    print(X)
    # perform a full pca
    # print(X.size)
    [D, W, mu] = fisherfaces(asRowMatrix(X), y)
    # import colormaps
    import matplotlib.cm as cm

    # turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in range(min(W.shape[1], 16)):
        e = W[:, i].reshape(X[0].shape)
        print(e.shape)
        E.append(normalize(e, 0, 255))
    # plot them and store the plot to "python_fisherfaces_fisherfaces.pdf"
    subplot(title="Fisherfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Fisherface", colormap=cm.gray,
            filename="python_fisherfaces_fisherfaces.png")

    print(W.shape)
    E = []
    for i in range(min(W.shape[1], 16)):
        e = W[:, i].reshape(-1, 1)
        P = project(e, X[0].reshape(1, -1), mu)
        R = reconstruct(e, P, mu)
        # reshape and append to plots
        R = R.reshape(X[0].shape)
        E.append(normalize(R, 0, 255))
    # plot them and store the plot to "python_reconstruction.pdf"
    subplot(title="Fisherfaces Reconstruction Yale FDB", images=E, rows=4, cols=4, sptitle="Fisherface",
            colormap=cm.gray, filename="python_fisherfaces_reconstruction.png")

    P = project(W, asRowMatrix(X), mu)

    print(P.shape)
    X_test, y_test = load_data('./Yale_Face_Database/Testing/')
    knn = KNN(k=1)
    knn.fit(P, y)
    predictions = knn.predict(project(W, asRowMatrix(X_test), mu))
    print(predictions)
    print(np.count_nonzero(predictions == y_test))
