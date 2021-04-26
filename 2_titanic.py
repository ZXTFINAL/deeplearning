import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv("train.csv")
dataset["Sex"] = dataset["Sex"].astype("category").cat.codes
dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
X = dataset[["Pclass", "Sex", "Fare", "Age"]].values
y = dataset["Survived"].values
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X-mean)/std
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n_train = X_train.shape[0]
n_features = X_train.shape[1]


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


w = np.random.rand(n_features)
b = 1.1


def model(x):
    z = w.dot(x.T)+b
    y_hat = sigmoid(z)
    return y_hat


epoches = 80000
lr = 0.01
for epoch in range(epoches):
    sum_w = np.zeros(n_features)
    sum_b = 0.0
    y_hat = model(X_train)
    sum_w = np.dot(y_hat-y_train, X_train)
    sum_b = np.sum(y_hat-y_train)
    grad_w = (1/n_train)*sum_w
    grad_b = (1/n_train)*sum_b
    w = w-lr*grad_w
    b = b-lr*grad_b

# for epoch in range(epoches):
#     sum_w = np.zeros(n_features)
#     sum_b = 0.0
#     for i in range(n_train):
#         xi = X_train[i]
#         yi_hat = model(xi)
#         yi = y_train[i]
#         sum_w += (yi_hat-yi)*xi
#         sum_b += (yi_hat-yi)
#     grad_w = (1/n_train)*sum_w
#     grad_b = (1/n_train)*sum_b
#     w = w-lr*grad_w
#     b = b-lr*grad_b


def predict(X):
    predictions = []
    n_samples = X.shape[0]

    for i in range(n_samples):
        xi = X[i]
        yi_hat = model(xi)
        if yi_hat < 0.5:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions


def get_accuracy(X, y):
    n_samples = X.shape[0]
    predictions = predict(X)
    loss = 0
    for i in range(n_samples):
        if y[i] != predictions[i]:
            loss += 1
    accuracy = (n_samples-loss)/n_samples
    return accuracy


train_accuracy = get_accuracy(X_train, y_train)
test_accuracy = get_accuracy(X_test, y_test)
print(train_accuracy)
print(test_accuracy)
