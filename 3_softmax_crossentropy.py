import numpy as np


def categorical_cross_entropy(y, y_hat):
    n_classes = len(y)
    loss = 0
    for i in range(n_classes):
        loss += -y[i]*np.log(y_hat[i])
    return loss


y = [0, 0, 1]
y_hat = [0.1, 0.1, 0.8]
loss = categorical_cross_entropy(y, y_hat)
print(loss)
y = [0, 0, 1]
y_hat = [0.1, 0.3, 0.6]
loss = categorical_cross_entropy(y, y_hat)
print(loss)
y = [0, 0, 1]
y_hat = [0.4, 0.5, 0.1]
loss = categorical_cross_entropy(y, y_hat)
print(loss)
