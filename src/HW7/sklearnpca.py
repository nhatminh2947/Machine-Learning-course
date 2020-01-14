from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import glob
import PIL.Image as Image
import numpy as np

rows = 60
cols = 60


def load_data(path):
    print('Loading... %s' % path)
    X = []
    y = []

    for id in range(1, 16):
        filelist = glob.glob(path + 'subject' + str(id).zfill(2) + "*")
        for fname in filelist:
            img = Image.open(fname)
            img = np.array(img.resize((rows, cols), Image.ANTIALIAS))

            X.append(img.flatten())
            y.append(id)
    print('There are %d samples' % len(y))
    return np.asarray(X), np.asarray(y)


X, y = load_data('./Yale_Face_Database/Training/')
X_test, y_test = load_data('./Yale_Face_Database/Testing/')

kpca = KernelPCA(n_components=25, kernel='linear')

X_transformed = kpca.fit_transform(X)

print(kpca.alphas_)
print(kpca.lambdas_)
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_transformed, y)
print(knn.predict(kpca.transform(X_test)))
print(y_test)

print(knn.score(kpca.transform(X_test), y_test))

print(X.shape)
lda = LinearDiscriminantAnalysis(solver='eigen', n_components=25)
X_transformed = lda.fit_transform(X, y)

print(lda.scalings_)
print(lda.scalings_.shape)
knn.fit(X_transformed, y)
print(knn.score(lda.transform(X_test), y_test))
