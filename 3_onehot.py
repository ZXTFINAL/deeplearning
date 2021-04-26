import numpy as np
n_classes = 3
y = np.array([0, 1, 0, 2, 1, 2, 2, 1, 0, 1])


def one_hot_encoding(labels, n_classes):
    result = np.eye(n_classes)[labels]
    return result


print(one_hot_encoding(y, n_classes))
