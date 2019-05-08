import torch
import math

class Module(object):

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
        print(self.x)
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

    def forward(self, *input):
        return []

    def backward(self, *gradwrtoutput):
        return[]

    def param(self):
        return[]


class Tanh(Module):

    def forward(self, *input):
        return []

    def backward(self, *gradwrtoutput):
        return[]

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