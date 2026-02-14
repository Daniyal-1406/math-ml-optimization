from .base import Optimizer


class GradientDescent(Optimizer):
    """
    Vanilla Gradient Descent:

        x_{k+1} = x_k - lr * grad
    """

    def step(self, params, grads):
        return params - self.lr * grads