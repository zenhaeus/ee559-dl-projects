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
    """Math for momentum see https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d """
    def __init__(self, params, lr=0.01, momentum=0):
        super(SGD, self).__init__(params)
        self.lr = lr
        self.momentum = momentum
        if (self.momentum != 0):
            self.Vt_prev_buffer = 0 # TODO find a good way to store previous gradients of each parameter (dictionary?)

    def step(self):
        """Perform a single optimization step"""
        for p, p_grad in self.params:
            Vt = p_grad
            if (self.momentum != 0):
                Vt = self.momentum*Vt_prev_buffer + (1-self.momentum)*Vt

            # update parameter
            p.add_(-self.lr*Vt)
