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

# 初始化模型参数
W = np.random.rand(10, 784)
b = np.zeros((10, 1))


def get_accuracy(X, y):
    n_samples = X.shape[0]
    y_hat = model(X)
    y_hat = np.argmax(y_hat, axis=1)
    y = np.argmax(y, axis=1)
    count = 0
    for i in range(len(y_hat)):
        if (y[i] == y_hat[i]):
            count += 1
    accuracy = count / n_samples
    return accuracy


def model(X):
    n_samples = X.shape[0]
    z = W.dot(X.T)+b
    exp_z = np.exp(z)
    sum_E = np.sum(exp_z, axis=0)
    y_hat = exp_z/sum_E
    return y_hat.T


epoches = 20000
lr = 0.05
for epoch in range(epoches):
    sum_w = np.zeros_like(W)
    sum_b = np.zeros_like(b)
    y_hat = model(X_train)
    sum_w = np.dot((y_hat-y_train).T, X_train)
    sum_b = np.sum((y_hat-y_train), axis=0).reshape((-1, 1))
    grad_w = (1/n_train)*sum_w
    grad_b = (1/n_train)*sum_b
    W = W-lr*grad_w
    b = b-lr*grad_b
    train_accuracy = get_accuracy(X_train, y_train)
    test_accuracy = get_accuracy(X_test, y_test)
    print("第"+str(epoch)+"次训练", train_accuracy, test_accuracy)
