from libsvm.commonutil import evaluations
from libsvm.svm import svm_problem, svm_parameter
from libsvm.svmutil import *
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import csv
import scipy

settings = ['-s 0 -t 1 -b 1 -q', '-s 0 -t 1 -b 1 -q', '-s 0 -t 1 -b 1 -q']

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

prob = svm_problem(y_train, x_train)
for setting in settings:
    print('Training setting {}'.format(setting))
    param = svm_parameter(setting)
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
    ACC, MSE, SCC = evaluations(y_test, p_label)
    print("ACC = {} MSE = {} SCC ={}".format(ACC, MSE, SCC))
    print('=====================================================')

