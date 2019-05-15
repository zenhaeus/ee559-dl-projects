from torch import empty as torch_empty
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

    # TODO: we do not need this function? Should be called in optimizer?
    def zero_grad(self):
        """Zeroes gradients of tensors which are to be optimized."""
        for param_pair in self.param():
            param_pair[1].zero_()

##########################################################

class Linear(Module):

    def __init__(self, nb_in, nb_out, bias=True):
        self.x = None
        self.s = None   # this does not need to be a class variable?

        # initialization with calibrated variance normal distribution
        # see: http://cs231n.github.io/neural-networks-2/
        epsilon = math.sqrt(2 / (nb_in + nb_out))
        self.weight = torch_empty(nb_out, nb_in).normal_(0, epsilon)
        self.weight_grad = torch_empty(nb_out, nb_in).zero_()

        if bias:
            self.bias = torch_empty(nb_out).normal_(0, epsilon)
            self.bias_grad = torch_empty(nb_out).zero_()
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
            param_list.extend(module.param())

        return param_list

##########################################################

class ReLU(Module):

    def __init__(self):
        self.x = None

    def forward(self, input):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        zeros = torch_empty(self.x.size()).zero_()
        return self.x.max(zeros)

    def backward(self, gradwrtoutput):
        gradwrtinput = (self.x > 0).float().mul(gradwrtoutput)
        self.x = None
        return gradwrtinput

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

##########################################################

class LossMSE(Module):

    def __init__(self):
        self.x = None
        self.target = None

    # FIXME: why is this here? Already defined in parent class?
    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input, target):
        assert self.x is None   # raise error if x has been defined before
        self.x = input
        self.target = target
        res = (self.x - self.target).pow(2).mean()
        return res

    def backward(self):
        gradwrtinput = 2*(self.x - self.target)
        gradwrtinput = gradwrtinput.div(self.x.shape[0])

        self.x = None
        self.target = None
        return gradwrtinput
