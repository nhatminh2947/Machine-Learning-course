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
cost = np.logspace(-3, 5, base=2, num=5)
gamma = np.logspace(-7, 3, base=2, num=6)
coef0 = [0, 1, 10]
degree = [2, 3, 4]
k_fold = 5

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

best_setting = None
best_accuracy = 0

for t in kernel_type:
    for c in cost:
        if t == 0:
            setting = '-s {} -t {} -c {} -v {} -b 1 -q'.format(s, t, c, k_fold)
            print('Training setting {}'.format(setting))
            param = svm_parameter(setting)
            acc = svm_train(prob, param)
            if best_accuracy < acc:
                best_accuracy = acc
                best_setting = setting
            print('=====================================================')
        elif t == 1:
            for d in degree:
                for r in coef0:
                    for g in gamma:
                        setting = '-s {} -t {} -d {} -r {} -c {} -g {} -v {} -b 1 -q'.format(s, t, d, r, c, g,
                                                                                             k_fold)
                        print('Training setting {}'.format(setting))
                        param = svm_parameter(setting)
                        acc = svm_train(prob, param)

                        if best_accuracy < acc:
                            best_accuracy = acc
                            best_setting = setting
                        print('=====================================================')
        elif t == 2:
            for g in gamma:
                setting = '-s {} -t {} -c {} -g {} -v {} -b 1 -q'.format(s, t, c, g, k_fold)
                print('Training setting {}'.format(setting))
                param = svm_parameter(setting)
                acc = svm_train(prob, param)

                if best_accuracy < acc:
                    best_accuracy = acc
                    best_setting = setting
                print('=====================================================')

print("Best accuracy: {}".format(best_accuracy))
print("Best setting: {}".format(best_setting))
