import glob
import PIL.Image as Image
import numpy as np
from pca_lda.decomposition import PCA
from pca_lda.KNN import KNN
import matplotlib.pyplot as plt

train_data_path = './Yale_Face_Database/Training/'
test_data_path = './Yale_Face_Database/Testing/'

rows = 231
cols = 195

D = rows * cols

print('d-dimension:', D)


def plot_eigenface(eig_vectors, shape, n=25):
    k = int(np.sqrt(n))
    f, axes = plt.subplots(k, k)
    for i in range(k):
        for j in range(k):
            axes[i, j].imshow(eig_vectors[:, i * k + j].reshape(shape), cmap='gray')
            axes[i, j].axis('off')

    plt.show()


def plot_reconstruct(X, shape):
    fig, axes = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            axes[i, j].imshow(X[i * 2 + j, :].reshape(shape), cmap='gray')
            axes[i, j].axis('off')

    plt.show()


def load_data(path):
    print('Loading... %s' % path)
    X = []
    y = []

    for id in range(0, 15):
        filelist = glob.glob(path + 'subject' + str(id + 1).zfill(2) + "*")
        for fname in filelist:
            img = np.array(Image.open(fname))
            X.append(img.reshape(rows * cols))
            y.append(id)

    print('There are %d samples' % len(y))
    return np.asarray(X), np.asarray(y)


if __name__ == '__main__':
    print('============================PCA============================')
    X_train, y_train = load_data(train_data_path)
    pca = PCA(X_train, n_components=25)

    # plot 25 eigenface
    # plot_eigenface(pca.eig_vectors, (rows, cols), n=25)

    # reconstruct 10 random image
    np.random.seed(2947)
    ids = np.random.randint(0, 135, size=10)
    X_transformed = pca.transform(X_train)

    X_reconstruct = pca.reconstruct(X_transformed[ids, :])
    plot_reconstruct(X_reconstruct, (rows, cols))

    # predict with knn
    X_test, y_test = load_data(test_data_path)

    knn = KNN(k=5)
    knn.fit(X_transformed, y_train)

    predictions = knn.predict(pca.transform(X_test), k_neighbors=5)
    true_positive = np.count_nonzero(predictions == y_test)
    false_positive = np.count_nonzero(predictions != y_test)
    precision = 1.0 * true_positive / (true_positive + false_positive)
    print('k_neighbors: %d precision: %.5f' % (5, precision))

    print('============================Kernel PCA RBF============================')
    X_train, y_train = load_data(train_data_path)
    kernel_pca = PCA(X_train, n_components=25, kernel='rbf', gamma=1e-9)
    X_train_transformed = kernel_pca.transform(X_train)

    knn = KNN(k=5)
    knn.fit(X_train_transformed, y_train)

    X_test, y_test = load_data(test_data_path)
    X_transformed_test = kernel_pca.transform(X_test)

    predictions = knn.predict(X_transformed_test, k_neighbors=5)
    true_positive = np.count_nonzero(predictions == y_test)
    false_positive = np.count_nonzero(predictions != y_test)
    precision = 1.0 * true_positive / (true_positive + false_positive)
    print('k_neighbors: %d precision: %.5f' % (5, precision))
    print('============================Kernel PCA poly============================')
    X_train, y_train = load_data(train_data_path)
    kernel_pca = PCA(X_train, n_components=25, kernel='poly', degree=3)
    X_train_transformed = kernel_pca.transform(X_train)

    knn = KNN(k=5)
    knn.fit(X_train_transformed, y_train)

    X_test, y_test = load_data(test_data_path)
    X_transformed_test = kernel_pca.transform(X_test)

    predictions = knn.predict(X_transformed_test, k_neighbors=1)
    true_positive = np.count_nonzero(predictions == y_test)
    false_positive = np.count_nonzero(predictions != y_test)
    precision = 1.0 * true_positive / (true_positive + false_positive)
    print('k_neighbors: %d precision: %.5f' % (5, precision))
    # X_train, y_train = load_data(train_data_path)
    # X_test, y_test = load_data(test_data_path)
    #
    # L = lda(X_train, y_train)
    #
    # X_transformed = lda.transform(X_train)
    # plot_eigenface(lda.w, (rows, cols), n=25)
    # ids = np.random.randint(0, 135, size=10)
    # # plot_reconstruct(lda.transform(X_train[ids, :].T), lda.mean_vector, (rows, cols))
    #
    # knn = KNN(k=1)
    # knn.fit(X_transformed, y_train)
    #
    # predictions = knn.predict(lda.transform(X_test))
    # true_positive = np.count_nonzero(predictions == y_test)
    # false_positive = np.count_nonzero(predictions != y_test)
    # precision = 1.0 * true_positive / (true_positive + false_positive)
    # print('k_neighbors: %d precision: %.5f' % (5, precision))
