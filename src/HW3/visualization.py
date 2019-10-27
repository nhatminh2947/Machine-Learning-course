import numpy as np
import matplotlib.pyplot as plt
a = 3
while 1:
    data = open('data.txt', 'r').read()
    lines = data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(' ')
            xs.append(float(x))
            ys.append(float(y))
    plt.scatter(xs, ys)

    data = open('weights.txt', 'r').read()
    lines = data.split("\n")
    n = int(lines[0])

    mean = np.array([[float(lines[i]) for i in range(1, n + 1)]]).T
    # print(mean)
    covar = []
    for i in range(n + 1, 2 * n + 1):
        if len(lines[i]) > 1:
            covar.append([float(var) for var in lines[i].strip().split(' ')])

    covar = np.array(covar)

    n_pred = 100
    x_pred = np.ones(n_pred)
    for i in range(1, n):
        x_pred = np.vstack((x_pred, np.linspace(-2, 2, n_pred) ** i))

    pred_mean = mean.T @ x_pred
    pred_var = np.sum(a + x_pred.T @ covar * x_pred.T, axis=1)

    # print(pred_mean)
    # print(pred_var)

    plt.plot(x_pred[1, :], pred_mean.T, 'r', label='Prediction')
    # Add predictive variance
    minus_var = pred_mean.flatten() - pred_var
    plus_var = pred_mean.flatten() + pred_var
    plt.fill_between(x_pred[1, :], minus_var, plus_var, alpha=0.1)
    plt.title('{:d} observations'.format(n + 1))
    plt.xlim((-2, 2))

    plt.legend()
    plt.pause(1)
    plt.clf()

plt.show()
