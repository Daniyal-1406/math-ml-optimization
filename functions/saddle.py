import numpy as np

def loss(x):
    return x[0]**2 - x[1]**2

def grad(x):
    return np.array([
        2 * x[0],
        -2 * x[1]
    ])