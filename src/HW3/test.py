import numpy as np
import matplotlib.pyplot as plt


def sample_true_model():
    # True parameters
    a = np.array([1, 2, 3, 4]).T
    sample_x = 2 * np.random.random() - 1
    sample_x = -0.736924
    x = np.array([[1, sample_x, sample_x ** 2, sample_x ** 3]]).T
    t = a.T @ x + np.random.normal(0, 1)
    t = -1.53226
    return x, t


alpha = 1
beta = 1

# Prior distribution parameters
m_0 = np.array([[0, 0, 0, 0]]).T
S_0 = 1 / alpha * np.identity(4)

n_points = 1
obs_x = []
obs_t = []

for n in range(n_points):
    # Get new data point
    x, t = sample_true_model()

    # Calculate posterior covariance and mean
    S_0_inv = np.linalg.inv(S_0)
    S_1 = np.linalg.inv(S_0_inv + beta * x @ x.T)
    m_1 = S_1 @ (S_0_inv @ m_0 + beta * t * x)

    # Plot observations
    obs_x.append(x[1])
    obs_t.append(t)
    plt.scatter(obs_x, obs_t, label='Observations')
    # Add predictive mean
    n_pred = 100
    x_pred = np.ones(n_pred)
    for i in range(1, 4):
        x_pred = np.vstack((x_pred, np.linspace(-2, 2, n_pred) ** i))
    # print(m_1.shape)
    # print(x_pred.shape)

    print(m_1)
    print(S_1)

    pred_mean = m_1.T @ x_pred
    pred_var = np.sum(1 / beta + x_pred.T @ S_1 * x_pred.T, axis=1)

    print(pred_mean)
    print(pred_var)

    plt.plot(x_pred[1, :], pred_mean.T, 'r', label='Prediction')
    # Add predictive variance
    minus_var = pred_mean.flatten() - pred_var
    plus_var = pred_mean.flatten() + pred_var
    plt.fill_between(x_pred[1, :], minus_var, plus_var, alpha=0.1)
    plt.title('{:d} observations'.format(n + 1))
    plt.xlim((-2, 2))
    plt.legend()

    plt.tight_layout()

    # For the next data point, the posterior will be a prior
    S_0 = S_1.copy()
    m_0 = m_1.copy()
    plt.pause(0.2)
    if n < n_points - 1:
        plt.clf()

plt.show()
