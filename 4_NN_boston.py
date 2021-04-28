from tensorflow.keras.datasets import boston_housing as bh
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
(X_train, y_train), (X_test, y_test) = bh.load_data()
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train-mean)/std
X_test = (X_test-mean)/std
print(X_train[0])
print(y_train[0])

model = Sequential()
model.add(Dense(64, input_shape=(13,), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation=None))
model.summary()
model.compile(optimizer=Adam(),
              loss="mse",
              metrics=["mae"]
              )
model.fit(X_train,
          y_train,
          epochs=300,
          batch_size=32,
          validation_split=0.2)
_, loss = model.evaluate(X_test, y_test)
print(loss)
