class Optimizer:
    """Base class for optimizers"""

    def __init__(self, params):
        self.params = params

    def step(self):
        """Performs optimization step."""
        raise NotImplementedError

    def zero_grad(self):
        """Zeroes gradients of tensors which are to be optimized."""
        for param in self.params:
            param[1].zero_()

class SGD(Optimizer):
    """Stochastic Gradient Descent Optimizer class."""

    def __init__(self, params, lr=0.01, momentum=0):
        super(SGD, self).__init__(params)
        self.lr = lr
        self.momentum = momentum

    def step(self):
        """Perform a single optimization step"""
        for param in self.params:
            param[0].add_(-self.lr*param[1])
