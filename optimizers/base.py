class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """
        Update parameters.
        """
        raise NotImplementedError("Step method must be implemented.")