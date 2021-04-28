import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

img = [ [3, 1, 1, 2, 8, 4],
        [1, 0, 7, 3, 2, 6], 
        [2, 3, 5, 1, 1, 3], 
        [1, 4, 1, 2, 6, 5], 
        [3, 2, 1, 3, 7, 2], 
        [9, 2, 6, 2, 5, 1]]
img= np.array(img).reshape(1, 6, 6, 1)
kernel = [[1, 0, -3],
          [6, 8, -1],
          [7, 6, 4]]
kernel = [np.array(kernel).reshape(3, 3, 1, 1)]

inputs = Input(shape=(6, 6, 1))
outputs = Conv2D(filters=1,
                 kernel_size=(3, 3),
                 strides=1,
                 padding="same",
                 use_bias=False,
                 weights=kernel)(inputs)
model = Model(inputs, outputs)
model.summary()