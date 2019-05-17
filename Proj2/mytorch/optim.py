import torch

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

        if (self.momentum != 0):
            self.Vt_prev_buffer = []
            for _, p_grad in self.params:
                self.Vt_prev_buffer.append(p_grad.clone())

    def step(self):
        """Perform a single optimization step"""
        for i, (p, p_grad) in enumerate(self.params):
            Vt = p_grad

            # momentum is implemented analogously to official PyTorch
            # see: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
            # note that this differs from Sutskever et. al.
            if (self.momentum != 0 and self.momentum is not None):
                Vt = self.momentum*self.Vt_prev_buffer[i] +  p_grad
                self.Vt_prev_buffer[i] = Vt

            # update parameter
            p.add_(-self.lr*Vt)
