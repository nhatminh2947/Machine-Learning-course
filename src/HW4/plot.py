import matplotlib.pyplot as plt
import numpy as np

gt = open('lr_data.txt', 'r').read()
lines = gt.split('\n')

X_red = np.empty((0, 2), float)
X_blue = np.empty((0, 2), float)
id_red = 0
id_blue = 0
for line in lines:
    if len(line) > 0:
        tmp = line.split(' ')

        if int(tmp[2]) == 0:
            X_red = np.append(X_red, [[float(tmp[0]), float(tmp[1])]], axis=0)
        else:
            X_blue = np.append(X_blue, [[float(tmp[0]), float(tmp[1])]], axis=0)
        id_blue += 1

plt.subplot(131)
plt.plot(X_red[:, 0], X_red[:, 1], "ro")
plt.plot(X_blue[:, 0], X_blue[:, 1], "bo")
plt.title("Ground truth")

gt = open('lr_gradient.txt', 'r').read()
lines = gt.split('\n')

X_red = np.empty((0, 2), float)
X_blue = np.empty((0, 2), float)
id_red = 0
id_blue = 0
for line in lines:
    if len(line) > 0:
        tmp = line.split(' ')

        if int(tmp[2]) == 0:
            X_red = np.append(X_red, [[float(tmp[0]), float(tmp[1])]], axis=0)
        else:
            X_blue = np.append(X_blue, [[float(tmp[0]), float(tmp[1])]], axis=0)
        id_blue += 1

plt.subplot(132)
plt.plot(X_red[:, 0], X_red[:, 1], "ro")
plt.plot(X_blue[:, 0], X_blue[:, 1], "bo")
plt.title("Gradient descent")

gt = open('lr_newton.txt', 'r').read()
lines = gt.split('\n')

X_red = np.empty((0, 2), float)
X_blue = np.empty((0, 2), float)
id_red = 0
id_blue = 0
for line in lines:
    if len(line) > 0:
        tmp = line.split(' ')

        if int(tmp[2]) == 0:
            X_red = np.append(X_red, [[float(tmp[0]), float(tmp[1])]], axis=0)
        else:
            X_blue = np.append(X_blue, [[float(tmp[0]), float(tmp[1])]], axis=0)
        id_blue += 1

plt.subplot(133)
plt.plot(X_red[:, 0], X_red[:, 1], "ro")
plt.plot(X_blue[:, 0], X_blue[:, 1], "bo")
plt.title("Newton's method")

plt.show()
