import numpy as np
import matplotlib.pyplot as plt



n_pred = 100
x_pred = np.vstack((np.ones(n_pred), np.linspace(-1, 1, n_pred)))
pred_mean = m_1.T @ x_pred
pred_var = np.sum(1 / beta + x_pred.T @ S_1 * x_pred.T, axis=1)
plt.plot(x_pred[1, :], pred_mean.T, 'r', label='Prediction')
# Add predictive variance
minus_var = pred_mean.flatten() - pred_var
plus_var = pred_mean.flatten() + pred_var
plt.fill_between(x_pred[1, :], minus_var, plus_var, alpha=0.1);
plt.title('{:d} observations'.format(n + 1))
plt.xlim((-1, 1))
plt.ylim((-1, 1))
plt.legend()
plt.show()