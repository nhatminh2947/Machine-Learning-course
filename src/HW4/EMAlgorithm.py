import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np
from math import log, isnan

mndata = MNIST('data')
w = 28
h = 28

converageCondition = 1e-5

images, labels = mndata.load_testing()
imagesT, labelsT = mndata.load_testing()
nData = len(images)
nTest = len(imagesT)
images = np.array(images)
# images = np.reshape(images,(nData,w,h))
labels = np.reshape(labels, (nData, 1))

imagesT = np.array(imagesT)
# imagesT = np.reshape(imagesT,(nTest,w,h))
labelsT = np.reshape(labelsT, (nTest, 1))

# filter data
images = []
labels = []
selLabels = [0, 2, 7]

K = len(selLabels)

print(str(K) + " Classes")
print(selLabels)
for i in range(nTest):
    if labelsT[i] in selLabels:  # or labelsT[i] == 3  ) :
        images.append(imagesT[i])
        labels.append(labelsT[i])

images = np.array(images)
labels = np.array(labels)

# nData = len(labels)
nData = 100
print(str(nData) + " data")
images = images[0:nData, ]
labels = labels[0:nData, ]

# CONVERT TO BINARY BIN
threshold = 256 / 2
images = images / threshold

# prob to see 1 at pixel i Pd[i]

# initial
pi = np.ones(K)
pi = (1.0 / K) * pi

mu = np.random.uniform(low=0.1, high=0.2, size=(K, (w * h)))

wei = np.zeros(K)
r = np.zeros((nData, K))  # responsibility
predictedLabel = [-1] * nData
predictedLabel = np.reshape(predictedLabel, (nData, 1))
accuracy = 0
countData = 1
countInteration = 0
countDataPerClass = np.zeros(K)

maxInteration = 1000
delta_change = 9999
outInteration = 0

mappingLabel = np.zeros(K)  # ppredicted label to real label

unique, counts = np.unique(labels, return_counts=True)

for inter in range(maxInteration):

    ith = 0
    print("E STEP")
    # print (str(pi) + " - " + str(counts*1.0/nData))
    for i in range(nData):
        Xi = np.reshape(images[i], (w * h, 1))
        for k in range(K):

            wei[k] = (pi[k])
            for d in range(w * h):
                wei[k] *= (mu[k][d] ** Xi[d]) * ((1 - mu[k][d]) ** (1 - Xi[d]))

        for k in range(K):
            r[i, k] = wei[k] / sum(wei)

    for i in range(nData):
        maxVal = -1
        selectedK = -1
        for k in range(K):
            if r[i, k] > maxVal:
                maxVal = r[i, k]
                selectedK = k
        predictedLabel[i] = selectedK

    print("M STEP")
    # M STEP
    N = np.zeros(K)

    # calculate Nk
    for k in range(K):
        for i in range(nData):
            N[k] += r[i, k]

    # update new Mu
    new_mu = np.zeros((K, (w * h)))

    # update new Mu
    for k in range(K):
        mean = np.zeros((1, w * h))
        for i in range(0, nData):
            X = np.reshape(images[i], (w * h, 1))
            mean += (r[i, k] * X.T)
        new_mu[k] = np.reshape(mean / N[k], (1, (w * h)))

    pi = N / nData

    delta_change = sum(sum(abs(new_mu - mu)))
    accuracy = 0
    # predictedLabel

    mappingLabel = np.zeros(K)  # ppredicted label to real label

    for k in range(K):
        countLabel = np.zeros(10)
        maxVal = -1
        selectedK = -1
        for i in range(nData):
            if predictedLabel[i] == k:
                countLabel[labels[i]] += 1

        for i in range(10):
            if countLabel[i] > maxVal:
                maxVal = countLabel[i]
                selectedK = i

        mappingLabel[k] = selectedK

    for i in range(nData):
        if mappingLabel[predictedLabel[i]] == labels[i]:
            accuracy += 1

    print("Current accuracy : " + str(accuracy * 100.0 / nData) + " %")

    if delta_change < converageCondition or (accuracy * 100.0 / nData) > 90:
        outInteration = inter
        break
    else:
        mu = new_mu

print("Data converage after " + str(outInteration + 1) + " interations ");

# calculate real Pi

print(mappingLabel)

unique, counts = np.unique(labels, return_counts=True)
print("Actual YES")
for i in range(K):
    print(str(unique[i]) + " :" + str(counts[i]))
print("Predicted YES")

uniquePredicted, countsPredicted = np.unique(predictedLabel, return_counts=True)
for i in range(K):
    print(str((int)(mappingLabel[uniquePredicted[i]])) + " :" + str(countsPredicted[i]))

# print ("Predicted - Real Label")
# for i in range (nData) :
# 	print ( str(mappingLabel[predictedLabel[i]]) + " - " + str (labels[i]))

# startistic

print("nData :" + str(nData))
for i in range(K):
    realLabel = mappingLabel[i]
    print("Class " + str(i))
    print("\tLabel \"" + str(int(mappingLabel[i])) + "\"")

    countTN = 0
    countFN = 0
    countFP = 0
    countTP = 0

    for d in range(nData):
        if predictedLabel[d] == i and labels[d] == realLabel:
            countTP += 1
        elif predictedLabel[d] != i and labels[d] != realLabel:
            countTN += 1
        elif predictedLabel[d] == i and labels[d] != realLabel:
            countFP += 1
        elif predictedLabel[d] != i and labels[d] == realLabel:
            countFN += 1

    print("\tTN : " + str(countTN))
    print("\tFN : " + str(countFN))
    print("\tTP : " + str(countTP))
    print("\tFP : " + str(countFP))

    print("Actual YES = FN + TP = " + str(countFN) + " + " + str(countTP) + " = " + str(countFN + countTP))
    print("Predicted YES = FN + TP = " + str(countFP) + " + " + str(countTP) + " = " + str(countFP + countTP))
    print("Actual NO = TN + FP = " + str(countTN) + " + " + str(countFP) + " = " + str(countTN + countFP))
    print("Predicted NO = TN + FN = " + str(countTN) + " + " + str(countFN) + " = " + str(countTN + countFN))
    print("Sensitivity : " + str(np.round(countTP * 100.0 / (countFN + countTP))) + "%")
    print("Specificity : " + str(np.round(countTN * 100.0 / (countTN + countFP))) + "%")
    print("")

visualize = True

if visualize:

    classify = []

    for i in range(nData):
        classify.append([int(predictedLabel[i]), np.amax(r[i,], axis=0)])

    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []
    c7 = []
    c8 = []
    c9 = []

    for i in range(nData):
        realLabel = mappingLabel[predictedLabel[i]]
        if realLabel == 0:
            c0.append([float(realLabel) + float((len(c0) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 1:
            c1.append([float(realLabel) + float((len(c1) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 2:
            c2.append([float(realLabel) + float((len(c2) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 3:
            c3.append([float(realLabel) + float((len(c3) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 4:
            c4.append([float(realLabel) + float((len(c4) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 5:
            c5.append([float(realLabel) + float((len(c5) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 6:
            c6.append([float(realLabel) + float((len(c6) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 7:
            c7.append([float(realLabel) + float((len(c7) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 8:
            c8.append([float(realLabel) + float((len(c8) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))
        elif realLabel == 9:
            c9.append([float(realLabel) + float((len(c9) * 0.01)), r[i, int(predictedLabel[i])]] / np.amax(r))

    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    c4 = np.array(c4)
    c5 = np.array(c5)
    c6 = np.array(c6)
    c7 = np.array(c7)
    c8 = np.array(c8)
    c9 = np.array(c9)

    print(c0.shape[0])
    print(c1.shape[0])
    print(c4.shape[0])

    if c0.shape[0] > 0:
        plt.plot(c0[:, 0], c0[:, 1], "bo")
    if c1.shape[0] > 0:
        plt.plot(c1[:, 0], c1[:, 1], "go")
    if c2.shape[0] > 0:
        plt.plot(c2[:, 0], c2[:, 1], "ro")
    if c3.shape[0] > 0:
        plt.plot(c3[:, 0], c3[:, 1], "co")
    if c4.shape[0] > 0:
        plt.plot(c4[:, 0], c4[:, 1], "mo")
    if c5.shape[0] > 0:
        plt.plot(c5[:, 0], c5[:, 1], "ko")
    if c6.shape[0] > 0:
        plt.plot(c6[:, 0], c6[:, 1], marker="o", color="darkred")
    if c7.shape[0] > 0:
        plt.plot(c7[:, 0], c7[:, 1], marker="o", color="limegreen")
    if c8.shape[0] > 0:
        plt.plot(c8[:, 0], c8[:, 1], marker="o", color="tomato")
    if c9.shape[0] > 0:
        plt.plot(c9[:, 0], c9[:, 1], marker="o", color="olive")

    plt.show()
