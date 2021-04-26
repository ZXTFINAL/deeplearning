from sklearn.preprocessing import MinMaxScaler
import numpy as np
X = [[-1, 2],
     [-0.5, 6],
     [0, 10],
     [1, 18]]
X = np.array(X)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X-X_min)/(X_max-X_min)
print(X_normalized)
# sklearn 的方法
sc = MinMaxScaler(feature_range=(0, 1))
X_normalized = sc.fit_transform(X)
print(X_normalized)
