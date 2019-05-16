import torch
import math
import matplotlib

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

# generate data set
def generate_disc_set(nb):
    """ Generate a disc data set """
    center = torch.Tensor([0.5, 0.5])
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = input.sub(center).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().div(-2).add(0.5).long()
    return input, target

def normalize_input_data(train_input, test_input):
    """ Normalize input data tensors """
    mean, std = train_input.mean(), train_input.std()

    train_input.sub(mean).div(std)
    test_input.sub(mean).div(std)
    return train_input, test_input

def generate_data_set(nb, normalize_inputs=True):
    """ Generate train and test data """
    train_input, train_target = generate_disc_set(nb)
    test_input, test_target = generate_disc_set(nb)

    if normalize_input_data:
        train_input, test_input = normalize_input_data(train_input, test_input)

    return train_input, train_target, test_input, test_target

def target_to_onehot(target):
    """ Convert target labels to one hot labels """
    res = torch.empty(target.size(0), 2).zero_()
    res.scatter_(1, target.view(-1, 1), 1.0).mul(0.9)
    return res

def compute_nb_errors(model, data_input, data_target):
    #output = model(data_input)
    #return (output != data_target).sum()
    pass





def plot_data_set(train_input, train_target):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        train_input[train_target == 0, 0],
        train_input[train_target == 0, 1],
        c = 'blue', s = 5
    )
    ax.scatter(
        train_input[train_target == 1, 0],
        train_input[train_target == 1, 1],
        c = 'red', s = 5
    )
    ax.axis([0, 1, 0, 1])
    ax.set_aspect('equal', 'box')
    ax.legend(
        ['class 0', 'class 1'], 
        loc = 'upper right', 
        fancybox = False, 
        framealpha = True
    )
    plt.show()
