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
lr = 1e-2
mini_batch_size = 1

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
        
        # train in minibatches
        # TODO : if mini_batch_size != 1, results differ from PyTorch implementation
        for k in range(0, int(train_input.size(0) / mini_batch_size)):
            # set gradients to zero
            optimizer.zero_grad()

            for b in range (mini_batch_size):
                # forward pass
                output = model(train_input[k+b])
                loss = criterion(output, train_target[k+b])
                    
                # backward pass
                model.backward(criterion.backward())
                sum_loss += loss.item()

            # one gradient step per minibatch
            optimizer.step()


        print('epoch: ', e, 'loss:', sum_loss)
#    print("Final output:\n{}".format(model(train_input)))

def mytorch_compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for k in range(data_input.size(0)):
        output = model(data_input[k])
        model.backward(output)  # this is only to free the saved input during forward pass
        predicted_class = torch.argmax(output)
        if data_target[k] != predicted_class:
            nb_data_errors = nb_data_errors + 1

    return nb_data_errors


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
mytorch_nb_train_errors = mytorch_compute_nb_errors(mytorch_model, train_input, train_target)

print('mytorch train errors: ', mytorch_nb_train_errors)

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

    criterion = nn.MSELoss(reduction = "mean")
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    for e in range(nb_epochs):
        sum_loss = 0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

        print('epoch: ', e, 'loss:', sum_loss)


def pytorch_compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        predicted_classes = torch.argmax(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

# define model using pytorch modules
pytorch_model = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
        )

# uniformly initialize all parameters to compare mytorch and pytorch
pytorch_weight_initialization(pytorch_model)

train_pytorch(pytorch_model, train_input, train_target_onehot)
pytorch_nb_train_errors = pytorch_compute_nb_errors(pytorch_model, train_input, train_target)

print('pytorch train errors: ', pytorch_nb_train_errors)