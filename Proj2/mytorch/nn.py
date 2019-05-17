from torch import empty as torch_empty
import math

class Module:

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def forward(self, input_):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


##########################################################

class Linear(Module):

    def __init__(self, nb_in, nb_out, bias=True):
        self.x = None
        self.s = None   # this does not need to be a class variable?

        # initialization with calibrated variance normal distribution
        # see: http://cs231n.github.io/neural-networks-2/
        std = math.sqrt(2 / (nb_in + nb_out))

        self.weight = torch_empty(nb_out, nb_in).normal_(0, std)
        self.weight_grad = torch_empty(nb_out, nb_in).zero_()

        if bias:
            self.bias = torch_empty(1, nb_out).normal_(0, std)
            self.bias_grad = torch_empty(1, nb_out).zero_()
        else:
            self.bias = None
            self.bias_grad = None

    def forward(self, input_):
        #assert self.x is None   # raise error if x has been defined before
        self.x = input_
        self.s = self.x.mm(self.weight.t())
        if self.bias is not None:
            self.s += self.bias
        return self.s

    def backward(self, gradwrtoutput):
        # gradient needs to be summed along 0-axis for bias, see https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html
        self.bias_grad += gradwrtoutput.sum(dim=0)
        self.weight_grad += self.x.t().mm(gradwrtoutput).t()
        self.x = None
        return gradwrtoutput.mm(self.weight)

    def param(self):
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]


##########################################################

class Sequential(Module):

    def __init__(self, *args):
        self.x = None

        self.module_list = []
        for module in args:
            self.module_list.append(module)

    def forward(self, input_):
        #assert self.x is None   # raise error if x has been defined before
        self.x = input_
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

    def forward(self, input_):
        #assert self.x is None   # raise error if x has been defined before
        self.x = input_
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

    def forward(self, input_):
        #assert self.x is None   # raise error if x has been defined before
        self.x = input_
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

    def forward(self, input_, target):
        #assert self.x is None   # raise error if x has been defined before
        self.x = input_
        self.target = target
        res = (self.x - self.target).pow(2).mean()
        return res

    def backward(self):
        gradwrtinput = 2*(self.x - self.target).div(self.x.shape[0])

        self.x = None
        self.target = None
        return gradwrtinput
