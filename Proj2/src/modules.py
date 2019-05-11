import torch
import math

class Module(object):

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])    # not sure if this is the good way to implement it?
 
    def forward (self, *input):
        raise NotImplementedError

    def backward (self, *gradwrtoutput):
        raise NotImplementedError

    def param (self):
        return []

##########################################################

class Linear(Module):

    def __init__(self, nb_in, nb_out, bias=True):
        self.x = None
        self.s = None

        # initialization
        epsilon = 1e-6
        self.weight = torch.empty(nb_out, nb_in).normal_(0, epsilon)
        self.weight_grad = torch.empty(nb_out, nb_in).zero_()

        if bias:
            self.bias = torch.empty(nb_out).normal_(0, epsilon)
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
        return torch.tanh(x)    # should we use the mathematical definition of tanh(x)?

    def backward(self, gradwrtoutput):
        gradwrtinput = (1 - self.x.tanh().pow(2)).mul(gradwrtoutput)
        self.x = None
        return gradwrtinput

    def param(self):
        return[]

##########################################################

class Sequential(Module):

    def forward(self, *input):
        return []

    def backward(self, *gradwrtoutput):
        return[]

    def param(self):
        return[]


##########################################################

class LossMSE(Module):

    def __init__(self):
        self.x = None
        self.target = None

    def forward(self, input, target):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        self.target = target
        return (self.target - self.x).pow(2).sum().div(self.x.shape[0])

    def backward(self, gradwrtoutput):
        # does it make sense to have an input argument "gradwrtoutput"?
        gradwrtinput = - (self.target - self.x).div(self.x.shape[0])
        self.x = None
        self.target = None
        return gradwrtinput

    def param(self):
        return[]