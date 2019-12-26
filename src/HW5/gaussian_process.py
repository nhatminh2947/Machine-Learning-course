import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from numpy.linalg import det


def read_file():
    x = []
    y = []
    with open('./input.data') as file:
        for line in file:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
    return np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)


def rational_quadratic_kernel(x, x_s, l=1.0, sigma=1.0, alpha=1.0):
    return (sigma ** 2) * (1 + cdist(x, x_s, 'sqeuclidean') / (2 * alpha * l ** 2)) ** (-alpha)


def plot_gp(mu, cov, x, x_train, y_train, fig_name):
    x = x.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.box(False)

    plt.figure(figsize=(8, 4))

    plt.fill_between(x, mu + uncertainty, mu - uncertainty, alpha=0.25, color='r', label='95% confidence interval')
    plt.plot(x, mu, label='Mean', color='r')

    plt.plot(x_train, y_train, 'bx', label='Training data')
    plt.legend()
    plt.title(fig_name)
    plt.savefig(fig_name + ".png")


def posterior_predictive(x_s, x_train, y_train, l=1.0, sigma_f=1.0, alpha=1.0, noise=1e-8):
    K = rational_quadratic_kernel(x_train, x_train, l, sigma_f, alpha) + noise * np.eye(len(x_train))
    K_s = rational_quadratic_kernel(x_train, x_s, l, sigma_f, alpha)
    K_ss = rational_quadratic_kernel(x_s, x_s, l, sigma_f, alpha) + noise
    K_inv = inv(K)
    mu_s = K_s.T.dot(K_inv).dot(y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def nll_fn(X, x_train, y_train):
    l = X[0]
    sigma = X[1]
    alpha = X[2]
    K = rational_quadratic_kernel(x_train, x_train, l, sigma, alpha)
    return 0.5 * np.log(det(K)) + 0.5 * y_train.T.dot(inv(K).dot(y_train)) + 0.5 * len(x_train) * np.log(2 * np.pi)


if __name__ == "__main__":
    x_train, y_train = read_file()
    noise = 1 / 5
    X = np.arange(-60, 60, 1).reshape(-1, 1)

    mu_s, cov_s = posterior_predictive(X, x_train, y_train, noise=noise)
    plot_gp(mu_s, cov_s, X, x_train=x_train, y_train=y_train, fig_name="Before optimize")

    res = minimize(nll_fn, x0=np.array([1, 1, 1]), args=(x_train, y_train),
                   bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')
    print(res.x)

    l_opt, sigma_f_opt, alpha_opt = res.x
    mu_s, cov_s = posterior_predictive(X, x_train, y_train, l=l_opt, sigma_f=sigma_f_opt, alpha=alpha_opt, noise=noise)
    plot_gp(mu_s, cov_s, X, x_train=x_train, y_train=y_train, fig_name="After optimize")
