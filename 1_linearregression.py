import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv("iris.data.csv")
# X = np.array(dataset["x3"])
# y = np.array(dataset["x4"])
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y = np.array([4, 7, 10, 13, 16, 19, 22, 24, 28, 32, 34, 36, 40, 42, 46])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True)
n_train = len(X_train)
n_test = len(X_test)
w = -0.3
b = 0.6
lr = 0.001
epoches = 5000


def model(x):
    y_hat = w*x+b
    return y_hat


for epoch in range(epoches):
    sum_w = 0.0
    sum_b = 0.0

    for i in range(n_train):
        xi = X_train[i]
        yi = y_train[i]
        yi_hat = model(xi)
        sum_w += (yi_hat-yi)*xi
        sum_b += (yi_hat-yi)
    grad_w = (2.0/n_train)*sum_w
    grad_b = (2.0/n_train)*sum_b

    w = w-lr*grad_w
    b = b-lr*grad_b
plt.rcParams["font.sans-serif"] = ["SimHei"]


def plots(w, b, X, y):
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.plot([i for i in range(0, 20)], [model(i) for i in range(0, 20)])
    plt.legend(("模型", "数据"), loc="upper left", prop={"size": 15})
    plt.title("线性回归模型", fontsize=15)
    plt.show()


plots(w, b, X, y)


def loss_funtion(X, y):
    total_loss = 0
    n_samples = len(X)
    for i in range(n_samples):
        xi = X[i]
        yi = y[i]
        yi_hat = model(xi)
        total_loss += (yi_hat - yi) ** 2
        avg_loss = (1 / n_samples) * total_loss
        return avg_loss


train_loss = loss_funtion(X_train, y_train)
test_loss = loss_funtion(X_test, y_test)
print(train_loss)
print(test_loss)
