import numpy as np
import matplotlib.pyplot as plt


def function(x):
    y = x**2+1
    return y


epoches = 50
lr = 0.1
xi = -18


def get_gradient(x):
    gradient = 2*x
    return gradient


trajectory = []


def get_x_star(xi):
    for i in range(epoches):
        trajectory.append(xi)
        xi = xi-lr*get_gradient(xi)
    x_star = xi
    return x_star


get_x_star(xi)
x = np.arange(-20, 20, 0.1)
y = function(x)

plt.plot(x, y)

x_trajectory = np.array(trajectory)
y_trajectory = function(np.array(trajectory))

plt.scatter(x_trajectory, y_trajectory)
plt.show()
