#!/usr/bin/env python
import torch
import mytorch.nn
import mytorch.data
import mytorch.optim
import mytorch.train

########################################################
# Toy data set

# create data
data = mytorch.data.generate_data_set(1000)
train_input, train_target, test_input, test_target = data
# create one hot encoding of training target for MSE loss
train_target_onehot = mytorch.data.target_to_onehot(train_target)

# check data set visually
#mytorch.data.plot_data_set(train_input, train_target)

########################################################
# Global training parameters
nb_epochs = 25
lr = 1e-1
mini_batch_size = 100
uniform_wi = True

#########################################################

# define model using mytorch modules
torch.set_grad_enabled(False)
mytorch_model = mytorch.nn.Sequential(
    mytorch.nn.Linear(2, 25), mytorch.nn.ReLU(),
    mytorch.nn.Linear(25, 25), mytorch.nn.ReLU(),
    mytorch.nn.Linear(25, 25), mytorch.nn.ReLU(),
    mytorch.nn.Linear(25, 25), mytorch.nn.ReLU(),
    mytorch.nn.Linear(25, 2)
)

# Classification using self written mytorch library
#myoptimizer = mytorch.optim.SGD(mytorch_model.param(), lr, 0.0)
mytorch_trainer = mytorch.train.MyTorchTrainer(mytorch_model, data, uniform_wi=uniform_wi)

mytorch_trainer.train(nb_epochs, mini_batch_size)
mytorch_trainer.print_summary()

#########################################################

# define model using pytorch modules
torch.set_grad_enabled(True)
pytorch_model = torch.nn.Sequential(
    torch.nn.Linear(2, 25), torch.nn.ReLU(),
    torch.nn.Linear(25, 25), torch.nn.ReLU(),
    torch.nn.Linear(25, 25), torch.nn.ReLU(),
    torch.nn.Linear(25, 25), torch.nn.ReLU(),
    torch.nn.Linear(25, 2)
)

# Classification comparison using PyTorch
#pyoptimizer = torch.optim.SGD(pytorch_model.parameters(), lr, 0.0)
pytorch_trainer = mytorch.train.PyTorchTrainer(pytorch_model, data, uniform_wi=uniform_wi)

pytorch_trainer.train(nb_epochs, mini_batch_size)
pytorch_trainer.print_summary()
