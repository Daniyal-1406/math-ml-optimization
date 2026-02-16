import numpy as np

def loss(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad(x):
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])