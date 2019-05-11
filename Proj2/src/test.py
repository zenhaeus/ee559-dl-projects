#!/usr/bin/env python

import torch
import modules
import math
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

torch.set_grad_enabled(False)

# generate data set
def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().div(-2).add(0.5).long()
    return input, target

def plot_data_set(train_input, train_target): 
    fig, ax = plt.subplots(1, 1)
    ax.scatter(train_input[train_target == 0, 0], train_input[train_target == 0, 1], c = 'blue', s = 5)
    ax.scatter(train_input[train_target == 1, 0], train_input[train_target == 1, 1], c = 'red', s = 5)
    ax.axis([0, 1, 0, 1])
    ax.set_aspect('equal', 'box')
    plt.show()

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

# check data set visually
#plot_data_set(train_input, train_target)


mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

# create one hot encoding for MSE loss
train_target_onehot = torch.empty(train_target.size(0), 2).zero_()
train_target_onehot.scatter_(1, train_target.view(-1, 1), 1.0).mul(0.9)

# define model
fc1 = modules.Linear(2, 128)
fc2 = modules.Linear(128, 2)
relu = modules.ReLU()
lossMSE = modules.LossMSE()


def forward_pass(input):
    x = fc1(input)
    x = relu(x)
    x = fc2(x)
    return x

def backward_pass():
    grad = lossMSE.backward(1)
    grad = fc2.backward(grad)
    grad = relu.backward(grad)
    grad = fc1.backward(grad)

epochs = 25
eta = 0.0001

# train
for e in range(0, epochs):    
    sum_loss = 0

    for k in range(0, train_input.size(0)):
        # forward pass
        output = forward_pass(train_input[k])
        loss = lossMSE(output, train_target_onehot[k])

        # backward pass
        backward_pass()

        # gradient descent
        #for p, p_grad in fc1.param():
        #    p =- eta * p_grad
        #for p, p_grad in fc2.param():
        #    p =- eta * p_grad
        fc1.weight -= eta * fc1.weight_grad
        fc1.bias -= eta * fc1.bias_grad
        fc2.weight -= eta * fc2.weight_grad
        fc2.bias -= eta * fc2.bias_grad

        sum_loss += loss.item()

    print('epoch: ', e, 'loss:', sum_loss)





