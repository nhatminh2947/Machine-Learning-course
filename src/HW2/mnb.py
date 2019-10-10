import numpy as np
from sklearn.naive_bayes import MultinomialNB

with open("train-images-idx3-ubyte", "rb") as file:
    magic_number = int.from_bytes(file.read(4), byteorder='big')
    n_images = int.from_bytes(file.read(4), byteorder='big')
    n_rows = int.from_bytes(file.read(4), byteorder='big')
    n_cols = int.from_bytes(file.read(4), byteorder='big')

    train_images = np.zeros((n_images, n_rows * n_cols))
    for i in range(n_images):
        for j in range(n_rows):
            for k in range(n_cols):
                train_images[i][j * 28 + k] = int.from_bytes(file.read(1), byteorder='big')

with open("train-labels-idx1-ubyte", "rb") as file:
    magic_number = file.read(4)
    n_images = int.from_bytes(file.read(4), byteorder='big')

    train_labels = np.zeros(n_images)
    for i in range(n_images):
        train_labels[i] = int.from_bytes(file.read(1), byteorder='big')

with open("t10k-images-idx3-ubyte", "rb") as file:
    magic_number = int.from_bytes(file.read(4), byteorder='big')
    n_images = int.from_bytes(file.read(4), byteorder='big')
    n_rows = int.from_bytes(file.read(4), byteorder='big')
    n_cols = int.from_bytes(file.read(4), byteorder='big')

    images = np.zeros((n_images, n_rows * n_cols))
    for i in range(n_images):
        for j in range(n_rows):
            for k in range(n_cols):
                images[i][j * 28 + k] = int.from_bytes(file.read(1), byteorder='big')

with open("t10k-labels-idx1-ubyte", "rb") as file:
    magic_number = file.read(4)
    n_images = int.from_bytes(file.read(4), byteorder='big')

    labels = np.zeros(n_images)
    for i in range(n_images):
        labels[i] = int.from_bytes(file.read(1), byteorder='big')


def extract_features(images, sz):
    features = np.zeros((sz, 32))
    print(features.shape)
    for i in range(sz):
        for pixel_id in range(28 * 28):
            pixel_value = images[i][pixel_id]
            features[i, int(pixel_value) // 8] += 1

    return features


X = extract_features(train_images, 60000)
X_test = extract_features(images, 10000)

print(X.shape)

clf = MultinomialNB()
clf.fit(X, train_labels)

print('predict: {}'.format(clf.predict(X_test[0:1])))
print('predict_log_proba: {}'.format(clf.predict_log_proba(X_test[0:1])))
print('predict_proba: {}'.format(clf.predict_proba(X_test[0:1])))