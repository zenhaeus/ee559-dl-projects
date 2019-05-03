import sys
import torch

from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn

from dlc_practical_prologue import *
from arguments import args

import torch.nn.functional as F

class Model(torch.nn.Module):
    """Class to encapsulate the creation, training and performance evaluation of the Model
    """

    def __init__(self, data=None, data_size=args.data_size, epochs=args.num_epochs, mini_batch_size=args.mini_batch_size):
        """ Initialize model object
        """
        super(Model, self).__init__()

        # Normalize inputs
        if data is None:
            data = generate_pair_sets(data_size)
        mu, std = data[0].mean(), data[0].std()
        self.train_input = data[0].sub(mu).div(std)
        self.test_input = data[3].sub(mu).div(std)

        self.train_target = data[1]
        self.train_classes = data[2]
        self.test_target = data[4]
        self.test_classes = data[5]
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self):
        """ Needs to be implemented in child class
        """
        pass

    def train(self):
        print("Train model: {}".format(self))
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-1)
        for e in range(1, self.epochs + 1):
            total_loss = 0
            for b in range(0, self.train_input.size(0), self.mini_batch_size):
                output = self(self.train_input.narrow(0, b, self.mini_batch_size))
                target = self.train_target.narrow(0, b, self.mini_batch_size)
                loss = self.criterion(output, target)
                self.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            cur_acc = 100 - self.test_error()
            cur_acc_train = 100 - self.train_error()
            self._update_progress(e / self.epochs, cur_acc, cur_acc_train)


    def train_error(self):
        return self.compute_nb_errors(self.train_input, self.train_target) / self.train_input.shape[0] * 100

    def test_error(self):
        return self.compute_nb_errors(self.test_input, self.test_target) / self.test_input.shape[0] * 100

    def compute_nb_errors(self, input, target):
        nb_errors = 0

        for b in range(0, input.size(0), self.mini_batch_size):
            output = self(input.narrow(0, b, self.mini_batch_size))
            _, predicted_targets = torch.max(output, 1)
            for k in range(self.mini_batch_size):
                if predicted_targets[k] != target[b + k]:
                    nb_errors = nb_errors + 1

        return nb_errors

    def _update_progress(self, progress, val_acc, train_acc):
        length = 20
        status = ""
        try:
            progress = float(progress)
        except:
            progress = 0
            status = "Error: progress must be numeric\r\n"

        if progress < 0:
            progress = 0
            status = "Error: progress must be >= 0\r\n"
        if progress >= 1:
            progress = 1
            status = "Finished\n"

        block = int(round(length * progress))
        text = "\rPercent: [{0}] {1}% Val Acc: {3:.1f}, Train Acc: {4:.1f} {2}".format(
            "#" * block + "-" * (length - block),
            round(progress * 100, 2),
            status,
            val_acc,
            train_acc
        )
        sys.stdout.write(text)
        sys.stdout.flush()


    def __str__(self):
        return "{0.__class__.__name__}(epochs={0.epochs}, mini_batch_size={0.mini_batch_size})".format(self)

class StupidNet1(Model):
    def __init__(self, nb_hidden):
        super(StupidNet1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),         # 32 x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 32 x 4 x 4
            nn.Conv2d(32, 64, kernel_size=3),        # 64 x 2 x 2
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 2 * 2, nb_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(nb_hidden, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 2 * 2)
        x = self.classifier(x)
        return x
