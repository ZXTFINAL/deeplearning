def RELU(z):
    if z<=0.0:
        return 0
    else:
        return z
print(RELU(-10))
print(RELU(8))