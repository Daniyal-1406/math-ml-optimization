import numpy as np
from .base import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp:

        s_{k+1} = rho * s_k + (1 - rho) * g_k^2
        x_{k+1} = x_k - epsilon / (sqrt(s_{k+1}) + delta) * g_k
    """

    def __init__(self, epsilon=0.01, rho=0.9, delta=1e-8):
        super().__init__(epsilon)
        self.rho = rho
        self.delta = delta
        self.s = None  # running average of squared gradients

    def step(self, params, grads):
        if self.s is None:
            self.s = np.zeros_like(params)

        self.s = self.rho * self.s + (1 - self.rho) * (grads ** 2)

        adjusted_grad = grads / (np.sqrt(self.s) + self.delta)

        return params - self.lr * adjusted_grad