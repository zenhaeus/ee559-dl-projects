#!/usr/bin/env python

import torch
import modules

torch.set_grad_enabled(False)

# testing some implementations

train_input = torch.tensor([1.0,2,3,4,5])
train_target = torch.tensor([10.0,11])

dl_do = torch.tensor([0.1, 0.2])

lin = modules.Linear(5,2)
relu = modules.ReLU()
lossMSE = modules.LossMSE()

epochs = 10
eta = 0.01

for e in range(0, epochs):    
    loss_tot = 0

    # forward pass
    x = lin(train_input)
    x = relu.forward(x)
    loss = lossMSE.forward(x, train_target)

    # backward pass
    lin.backward(relu.backward(lossMSE.backward(1)))

    # gradient descent
    for p, p_grad in lin.param():
        p =- eta * p_grad
        print(p, p_grad)

    loss_tot += loss
    print('epoch: ', e, 'loss:', loss_tot)





