import numpy as np


def softmax(array):
    t = np.exp(array)
    s = np.sum(t)
    result = t/s
    return result


a = np.array([1, 3, 5])
result = softmax(a)
print(result)
print(sum(result))
