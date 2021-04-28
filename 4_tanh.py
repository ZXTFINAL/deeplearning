import numpy as np


def tanh(z):
    y = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return y


print(tanh(0))
print(tanh(1))
print(tanh(-1))
