import numpy as np
from .base import Optimizer


class Momentum(Optimizer):
    """
    Heavy Ball Momentum:

        v_{k+1} = beta * v_k + grad
        x_{k+1} = x_k - lr * v_{k+1}
    """

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None  # velocity

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.beta * self.v + grads
        return params - self.lr * self.v