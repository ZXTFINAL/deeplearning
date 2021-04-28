from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=3,
                input_shape=(2,),
                activation="sigmoid",
                kernel_initializer="ones",
                bias_initializer="zeros"))
model.add(Dense(units=2,
                activation="softmax",
                kernel_initializer="ones",
                bias_initializer="zeros"))
model.summary()
