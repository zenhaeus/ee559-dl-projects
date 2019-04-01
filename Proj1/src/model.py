import torch
from torch import nn

class Model(torch.nn.Module):
    """Class to encapsulate the creation, training and performance evaluation of the Model
    """

    def __init__(self, data, mini_batch_size):
        """ Initialize model object
        """
        super(Model, self).__init__()
        self.train_input = data[0]
        self.train_target = data[1]
        self.train_classes = data[2]
        self.test_input = data[3]
        self.test_target = data[4]
        self.test_classes = data[5]
        self.mini_batch_size = mini_batch_size
        self.criterion = nn.MSELoss

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward(self):
        pass

    def train(self):
        criterion = self.criterion
        eta = 1e-1

        for e in range(25):
            sum_loss = 0
            for b in range(0, self.train_input.size(0), self.mini_batch_size):
                output = self(self.train_input.narrow(0, b, self.mini_batch_size))
                loss = criterion(output, self.train_target.narrow(0, b, self.mini_batch_size))
                self.zero_grad()
                loss.backward()
                sum_loss = sum_loss + loss.item()
                for p in self.parameters():
                    p.data.sub_(eta * p.grad.data)
            print(e + 1, sum_loss)


    def train_error(self):
        return self.compute_nb_errors(self.train_input, self.train_target)

    def test_error(self):
        return self.compute_nb_errors(self.test_input, self.test_target)

    def compute_nb_errors(self, input, target):
        nb_errors = 0

        for b in range(0, input.size(0), self.mini_batch_size):
            output = self(input.narrow(0, b, self.mini_batch_size))
            _, predicted_targets = torch.max(output, 1)
            for k in range(self.mini_batch_size):
                if predicted_targets[k] != target[b + k]:
                    nb_errors = nb_errors + 1

        return nb_errors
