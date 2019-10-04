import matplotlib.pyplot as plt
import numpy as np


def f(x, coeffs):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i] * (x ** i)
    return y


with open("testfile.txt") as file:
    lines = file.readlines()
    data = np.array([[float(x) for x in line.strip().split(',')] for line in lines])

with open("coeffs.txt") as file:
    lines = file.readlines()
    coefficients = np.array([float(coeff.strip()) for coeff in lines])

x = np.arange(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 0.01)
y = f(x, coefficients)

print()

fig, ax = plt.subplots(figsize=(12, 4))
# plt.figure(figsize=(10, 20))
ax.scatter(data[:, 0], data[:, 1], c='R')

ax.plot(x, y)

ax.set(xlabel='x', ylabel='y', title='Using LSE')
ax.grid()

fig.savefig("result.png")
plt.show()
