import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


x = np.arange(-10, 10, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.scatter(0, sigmoid(0))
plt.show()
