#!/usr/bin/env python
import torch
import mytorch.nn
import mytorch.data
import mytorch.optim

torch.set_grad_enabled(False)

# Create data
train_input, train_target, test_input, test_target = mytorch.data.generate_data_set(1000)
# create one hot encoding of training target for MSE loss
train_target_onehot = mytorch.data.target_to_onehot(train_target)

# check data set visually
#plot_data_set(train_input, train_target)


# define model
lossMSE = mytorch.nn.LossMSE()
model = mytorch.nn.Sequential(
    mytorch.nn.Linear(2, 128),
    mytorch.nn.Linear(128, 2),
    mytorch.nn.ReLU()
)

nb_epochs = 10
lr = 1e-4

# train with self written autograd
def train_mytorch(nb_epochs, model):
    print('Self written autograd  -------')
    optimizer = mytorch.optim.SGD(model.param(), lr)
    for e in range(0, nb_epochs):
        sum_loss = 0

        for k in range(0, train_input.size(0)):
            # forward pass
            output = model(train_input[k])
            loss = lossMSE(output, train_target_onehot[k])

            # set gradients to zero
            model.zero_grad()

            # backward pass
            model.backward(lossMSE.backward())
            #backward_pass()
            optimizer.step()

            sum_loss += loss.item()

        print('epoch: ', e, 'loss:', sum_loss)
#    print("Final output:\n{}".format(model(train_input)))

train_mytorch(nb_epochs, model)

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
#with torch.no_grad():
#    for p in model.parameters():
#        #p.normal_(0, 1e-6)
#        p.fill_(1e-6)

train_model(model, train_input, train_target_onehot)
