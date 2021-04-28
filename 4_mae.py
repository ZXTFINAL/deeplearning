import numpy as np
def mean_square_error(y, y_hat):
    # 第一种
    return np.mean(np.square(np.array(y)-np.array(y_hat)))
    # 第二种
    # length = len(y)
    # loss = 0
    # for i, j in zip(y, y_hat):
    #     loss += np.square(i-j)
    # return loss/length
    # 第三种
    # return np.mean([np.square(i-j) for i, j in zip(y, y_hat)])
y = [5.6, 9.6, 1.3]
y_hat = [2.5, 4.1, 5.8]
loss = mean_square_error(y, y_hat)
print(loss)
y = [5.6, 9.6, 1.3]
y_hat = [5.2, 8.4, 0.9]
loss = mean_square_error(y, y_hat)
print(loss)
        