import numpy as np
image = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
image = np.array(image)
image = image.reshape((1, 9))
print(image)
