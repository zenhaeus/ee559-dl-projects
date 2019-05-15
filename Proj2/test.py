#!/usr/bin/env python
import torch
import mytorch.nn
import mytorch.data
import mytorch.optim

########################################################
# Toy data set

# create data
train_input, train_target, test_input, test_target = mytorch.data.generate_data_set(1000)
# create one hot encoding of training target for MSE loss
train_target_onehot = mytorch.data.target_to_onehot(train_target)

# check data set visually
#mytorch.data.plot_data_set(train_input, train_target)

########################################################
# Global training parameters
nb_epochs = 10
lr = 1e-5

#########################################################
# Classification using self written mytorch library

# globally disable automatic differentiation of PyTorch
torch.set_grad_enabled(False)

def mytorch_weight_initialization(model):
    for p, _ in mytorch_model.param():
        p.fill_(1e-6)

def train_mytorch(model, train_input, train_target):
    print('Training of mytorch model  -------')

    criterion = mytorch.nn.LossMSE()
    optimizer = mytorch.optim.SGD(model.param(), lr, momentum = 0.9)
    
    for e in range(0, nb_epochs):
        sum_loss = 0
        
        for k in range(0, train_input.size(0)):
            # forward pass
            output = model(train_input[k])
            loss = criterion(output, train_target[k])

            # set gradients to zero
            optimizer.zero_grad()
                
            # backward pass
            model.backward(criterion.backward())

            optimizer.step()

            sum_loss += loss.item()

        print('epoch: ', e, 'loss:', sum_loss)
#    print("Final output:\n{}".format(model(train_input)))

# define model using mytorch modules
mytorch_model = mytorch.nn.Sequential(
        mytorch.nn.Linear(2, 128),
        mytorch.nn.ReLU(),
        mytorch.nn.Linear(128, 2)
        )   

#print(mytorch_model.module_list)
#print(mytorch_model.param())

# uniformly initialize all parameters to compare mytorch and pytorch
mytorch_weight_initialization(mytorch_model)

train_mytorch(mytorch_model, train_input, train_target_onehot)

#########################################################
# Classification comparison using PyTorch

from torch import nn
from torch import optim
torch.set_grad_enabled(True)

def pytorch_weight_initialization(model):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1e-6)

def train_pytorch(model, train_input, train_target):
    print('Training of pytorch model  -------')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    mini_batch_size = 1

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

# define model using pytorch modules
pytorch_model = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
        )

# uniformly initialize all parameters to compare mytorch and pytorch
pytorch_weight_initialization(pytorch_model)

train_pytorch(pytorch_model, train_input, train_target_onehot)
