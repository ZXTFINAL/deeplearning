import numpy as np


def binary_cross_entropy(y, y_hat):
    loss = -y*np.log(y_hat)-(1-y)*np.log(1-y_hat)
    return loss


print(binary_cross_entropy(0, 0.01))
print(binary_cross_entropy(1, 0.99))
print(binary_cross_entropy(0, 0.80))
print(binary_cross_entropy(1, 0.20))
