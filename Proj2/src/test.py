#!/usr/bin/env python

import torch
import modules

# testing some implementations

train_input = torch.tensor([1.0,2,3,4,5])
train_target = torch.tensor([10.0,11])

dl_do = torch.tensor([0.1, 0.2])

lin = modules.Linear(5,2)

print("Forward: ", lin.forward(train_input))
print("Backward: ", lin.backward(dl_do))
print("Param: ", lin.param())


