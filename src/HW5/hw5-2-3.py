from libsvm.commonutil import evaluations
from libsvm.svm import svm_problem, svm_parameter
from libsvm.svmutil import *
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import csv
import scipy

settings = ['-s 0 -t 0 -c 4 -b 1 -v 10 -q', '-s 0 -t 1 -c 4 -b 1 -v 10 -q', '-s 0 -t 2 -c 4 -b 1 -v 10 -q']

s = 0
kernel_type = [0, 1, 2]
cost = [-9, 11, 2]
gamma = [-15, 5, 2]
k_fold = 10

with open('./X_train.csv') as x_train_file:
    csv_reader = csv.reader(x_train_file, delimiter=',')
    x_train = scipy.asarray([row for row in csv_reader], dtype=float)

with open('./Y_train.csv') as x_train_file:
    csv_reader = csv.reader(x_train_file, delimiter=',')
    y_train = scipy.asarray([row for row in csv_reader], dtype=float).squeeze()

with open('./X_test.csv') as x_train_file:
    csv_reader = csv.reader(x_train_file, delimiter=',')
    x_test = scipy.asarray([row for row in csv_reader], dtype=float)

with open('./Y_test.csv') as x_train_file:
    csv_reader = csv.reader(x_train_file, delimiter=',')
    y_test = scipy.asarray([row for row in csv_reader], dtype=float).squeeze()


def custom_kernel(x_train, x_test):
    gamma = 0.03125
    train_linear_kernel = np.matmul(x_train, np.transpose(x_train))
    train_rbf_kernel = squareform(np.exp(-gamma * pdist(x_train, 'sqeuclidean')))
    x_train_kernel = np.hstack((np.arange(1, 5001).reshape((5000, 1)), np.add(train_linear_kernel, train_rbf_kernel)))

    test_linear_kernel = np.matmul(x_test, np.transpose(x_train))
    test_rbf_kernel = np.exp(-gamma * cdist(x_test, x_train, 'sqeuclidean'))
    x_test_kernel = np.hstack((np.arange(1, 2501).reshape((2500, 1)), np.add(test_linear_kernel, test_rbf_kernel)))

    return x_train_kernel, x_test_kernel


def svm(x_train_kernel, y_train, x_test_kernel, y_test):
    prob = svm_problem(y_train, x_train_kernel, isKernel=True)
    param = svm_parameter('-q -t 4 -c 32')
    model = svm_train(prob, param)
    svm_predict(y_test, x_test_kernel, model)


if __name__ == "__main__":
    x_train_kernel, x_test_kernel = custom_kernel(x_train, x_test)
    svm(x_train_kernel, y_train, x_test_kernel, y_test)
