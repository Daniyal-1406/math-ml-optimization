import numpy as np


def make_quadratic(condition_number=10):
    """
    Creates quadratic function:
        f(x) = 1/2 x^T A x
    A = diag(1, condition_number)
    """

    A = np.array([[1.0, 0.0],
                  [0.0, float(condition_number)]])

    def loss(x):
        return 0.5 * x.T @ A @ x

    def grad(x):
        return A @ x

    return loss, grad, A