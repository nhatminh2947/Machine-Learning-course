import matplotlib.pyplot as plt
import numpy as np

w = []
a = 1
obs = [10, 50, 10000]

gt = open('weights_0.txt', 'r').read()
lines = gt.split('\n')
for line in lines:
    if len(line) > 0:
        w.append(float(line.strip()))

a = w[-1]
w = np.array([w[:-1]]).T
beta = 1 / a
n = len(w)

n_pred = 1000
x_pred = np.ones(n_pred)
for i in range(1, n):
    x_pred = np.vstack((x_pred, np.linspace(-2, 2, n_pred) ** i))

pred_mean = w.T @ x_pred
pred_var = np.full(pred_mean.shape, np.sqrt(a))
plt.plot(x_pred[1, :], pred_mean.T, 'k')
minus_var = pred_mean.flatten() - pred_var.flatten()
plus_var = pred_mean.flatten() + pred_var.flatten()
plt.plot(x_pred[1, :], minus_var, 'r')
plt.plot(x_pred[1, :], plus_var, 'r')

plt.title('Ground truth')
plt.xlim((-2, 2))
plt.ylim((-30, 30))
plt.show()

data = open('data.txt', 'r').read()
lines = data.split('\n')
xs = []
ys = []
for line in lines:
    if len(line) > 0:
        x, y = line.split(' ')
        xs.append(float(x))
        ys.append(float(y))

for n_obs in obs:
    plt.scatter(xs[:n_obs], ys[:n_obs])

    data = open('weights_{}.txt'.format(n_obs), 'r').read()
    lines = data.split("\n")

    mean = np.array([[float(lines[i]) for i in range(1, n + 1)]]).T
    # print(mean)
    covar = []
    for i in range(n + 1, 2 * n + 1):
        if len(lines[i]) > 0:
            covar.append([float(var) for var in lines[i].strip().split(' ')])

    covar = np.array(covar)

    n_pred = 1000
    x_pred = np.ones(n_pred)
    for i in range(1, n):
        x_pred = np.vstack((x_pred, np.linspace(-2, 2, n_pred) ** i))

    pred_mean = mean.T @ x_pred
    pred_var = np.sum(1 / beta + x_pred.T @ covar * x_pred.T, axis=1)
    pred_std = np.sqrt(pred_var)

    # print(pred_mean)
    # print(pred_var)

    plt.plot(x_pred[1, :], pred_mean.T, 'k')
    # Add predictive variance
    minus_var = pred_mean.flatten() - pred_std
    plus_var = pred_mean.flatten() + pred_std
    plt.plot(x_pred[1, :], minus_var, 'r')
    plt.plot(x_pred[1, :], plus_var, 'r')

    plt.title('{:d} observations'.format(n_obs))
    plt.xlim((-2, 2))
    plt.ylim((-30, 30))
    plt.show()
