import numpy as np
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


def one_hot_encoding(labels, n_classes):
    result = np.eye(n_classes)[labels]
    return result


n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_classes = 10
flatten_size = 28*28
X_train = X_train/255
X_train = X_train.reshape((n_train, flatten_size))
y_train = one_hot_encoding(y_train, n_classes)
X_test = X_test/255
X_test = X_test.reshape((n_test, flatten_size))
y_test = one_hot_encoding(y_test, n_classes)
