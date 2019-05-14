#!/usr/bin/env python

import torch
import mytorch.nn
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
fc1 = mytorch.nn.Linear(2, 128)
fc2 = mytorch.nn.Linear(128, 2)
relu = mytorch.nn.ReLU()
lossMSE = mytorch.nn.LossMSE()

# test sequential module
seq = mytorch.nn.Sequential(fc1, relu, fc2)

def forward_pass(input):
    #x = fc1(input)
    #x = relu(x)
    #x = fc2(x)
    return seq.forward(input)

def backward_pass():
    grad = lossMSE.backward()
    #grad = fc2.backward(grad)
    #grad = relu.backward(grad)
    #grad = fc1.backward(grad)
    grad = seq.backward(grad)

nb_epochs = 10
lr = 1e-4

# train with self written autograd
print('Self written autograd  -------')
for e in range(0, nb_epochs):
    sum_loss = 0

    for k in range(0, train_input.size(0)):
        # forward pass
        output = forward_pass(train_input[k])
        loss = lossMSE(output, train_target_onehot[k])

        # set gradients to zero
        fc1.weight_grad.zero_()
        fc1.bias_grad.zero_()
        fc2.weight_grad.zero_()
        fc2.bias_grad.zero_()

        # backward pass
        backward_pass()

        # gradient descent
        # make this somehow work with sequential module...

        #for p, p_grad in fc1.param():
        #    p =- lr * p_grad
        #for p, p_grad in fc2.param():
        #    p =- lr * p_grad
        #fc1.weight -= lr * fc1.weight_grad
        #fc1.bias -= lr * fc1.bias_grad
        #fc2.weight -= lr * fc2.weight_grad
        #fc2.bias -= lr * fc2.bias_grad

        sum_loss += loss.item()

    print('epoch: ', e, 'loss:', sum_loss)



# check with normal PyTorch
from torch import nn
from torch import optim
torch.set_grad_enabled(True)

mini_batch_size = 1

def create_shallow_model():
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )

def train_model(model, train_input, train_target):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr)

    for e in range(nb_epochs):
        sum_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        print('epoch: ', e, 'loss:', sum_loss)


print('PyTorch autograd ------- ')
model = create_shallow_model()

# make sure PyTorch model is also initialized with normal distribution
with torch.no_grad():
    for p in model.parameters():
        #p.normal_(0, 1e-6)
        p.fill_(1e-6)

train_model(model, train_input, train_target_onehot)
