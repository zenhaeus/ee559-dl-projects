import torch
import math

class Module:

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward (self, *input):
        raise NotImplementedError

    def backward (self, *gradwrtoutput):
        raise NotImplementedError

    def param (self):
        return []

    def zero_grad(self):
        """Zeroes gradients of tensors which are to be optimized."""
        # TODO: write test
        for param in self.param():
            for p in param:
                p[1].zero_()

##########################################################

class Linear(Module):

    def __init__(self, nb_in, nb_out, bias=True):
        self.x = None
        self.s = None   # this does not need to be a class variable?

        # initialization
        epsilon = 1e-6
        self.weight = torch.empty(nb_out, nb_in).normal_(0, epsilon)
        # FIXME: why are the weights overriden with epsilon below?
        self.weight.fill_(epsilon)
        self.weight_grad = torch.empty(nb_out, nb_in).zero_()

        if bias:
            self.bias = torch.empty(nb_out).normal_(0, epsilon)
            # FIXME: why are the biases overriden with epsilon below?
            self.bias.fill_(epsilon)
            self.bias_grad = torch.empty(nb_out).zero_()
        else:
            self.bias = None
            self.bias_grad = None

    def forward(self, input):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        self.s = self.weight.mv(self.x)
        if self.bias is not None:
            self.s += self.bias
        return self.s

    def backward(self, gradwrtoutput):
        self.bias_grad += gradwrtoutput
        self.weight_grad += gradwrtoutput.view(-1, 1).mm(self.x.view(1, -1))
        gradwrtinput = self.weight.t().mv(gradwrtoutput)
        self.x = None
        return gradwrtinput

    def param(self):
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]


##########################################################

class Sequential(Module):

    def __init__(self, *args):
        self.x = None

        self.module_list = []
        for module in args:
            self.module_list.append(module)

    def forward(self, input):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        output = self.x

        # go through all modules and calculate outputs sequentially
        for module in self.module_list:
            output = module.forward(output)

        return output

    def backward(self, gradwrtoutput):
        gradwrtinput = gradwrtoutput
        # go through all modules in reversed order and backpropagate sequentially
        for module in reversed(self.module_list):
            gradwrtinput = module.backward(gradwrtinput)

        self.x = None
        return gradwrtinput

    def param(self):
        param_list = []
        for module in self.module_list:
            param_list.append(module.param())

        return param_list

##########################################################

class ReLU(Module):

    def __init__(self):
        self.x = None

    def forward(self, input):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        return torch.max(self.x, torch.zeros(self.x.shape))

    def backward(self, gradwrtoutput):
        gradwrtinput = (self.x > 0).float().mul(gradwrtoutput)
        self.x = None
        return gradwrtinput

    def param(self):
        return[]

##########################################################

class Tanh(Module):

    def __init__(self):
        self.x = None

    def forward(self, input):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        return self.x.tanh()

    def backward(self, gradwrtoutput):
        gradwrtinput = (1 - self.x.tanh().pow(2)).mul(gradwrtoutput)
        self.x = None
        return gradwrtinput

    def param(self):
        return[]

##########################################################

class LossMSE(Module):

    def __init__(self, method='mean'):
        self.x = None
        self.target = None
        self.method = method

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input, target):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        self.target = target
        res = (self.x - self.target).pow(2)
        if self.method == 'mean': # FIXME: only consider mean?
            return res.mean()
        else:
            return res.sum()

    def backward(self):
        # FIXME: do we need an argument "gradwrtoutput"?
        # FIXME: take mean / sum into account?
        gradwrtinput = 2*(self.x - self.target).div(self.x.shape[0])
        self.x = None
        self.target = None
        return gradwrtinput

    def param(self):
        return[]
