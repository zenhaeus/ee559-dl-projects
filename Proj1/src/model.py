import sys
import abc
import time

import torch
from torch import nn
from torch import optim

from dlc_practical_prologue import *
from arguments import args


class Model(torch.nn.Module, metaclass=abc.ABCMeta):
    """Encapsulates creation, training and evaluation of a model."""

    def __init__(
        self, data_size=args.data_size, epochs=args.num_epochs,
        mini_batch_size=args.mini_batch_size
    ):
        """Initialize a model object."""
        super(Model, self).__init__()

        self.data_size = data_size
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

        self.acc_train = []
        self.acc_val = []
        self.time = []

    def data(self):
        """Randomly samples and normalizes data."""
        data = generate_pair_sets(self.data_size)

        # Normalize inputs
        mu, std = data[0].mean(), data[0].std()
        self.train_input = data[0].sub(mu).div(std)
        self.test_input = data[3].sub(mu).div(std)

        self.train_target = data[1]
        self.train_classes = data[2]
        self.test_target = data[4]
        self.test_classes = data[5]

    def weight_reset(self):
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()

    def count_params(self):
        ps = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.tensor(p.size()).prod().item() for p in ps])

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    @abc.abstractmethod
    def forward(self):
        """Needs to be implemented in a child class."""
        return

    def train(self):
        time_start = time.time()

        # Initialise new random data and reset weights
        self.data()
        self.weight_reset()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        for e in range(1, self.epochs + 1):
            total_loss = 0
            for b in range(0, self.train_input.size(0), self.mini_batch_size):
                output = self(self.train_input.narrow(
                    0, b, self.mini_batch_size))
                target = self.train_target.narrow(0, b, self.mini_batch_size)

                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            cur_acc_val = 100 - self.test_error()
            cur_acc_train = 100 - self.train_error()
            self._update_progress(e / self.epochs, cur_acc_val, cur_acc_train)
        self.acc_train.append(cur_acc_train)
        self.acc_val.append(cur_acc_val)
        time_end = time.time()
        self.time.append(time_end - time_start)

    def train_multiple(self, n=3):
        print("Train {} times".format(n))
        print("Model: {}".format(self))
        print("N params: {}".format(self.count_params()))
        self.acc_train, self.acc_train = [], []

        for i in range(n):
            self.train()

        self.acc_val = torch.tensor(self.acc_val)
        self.acc_train = torch.tensor(self.acc_train)
        self.time = torch.tensor(self.time)

        mean_val, std_val = self.acc_val.mean(), self.acc_val.std()
        mean_train, std_train = self.acc_train.mean(), self.acc_train.std()
        mean_time, std_time = self.acc_train.mean(), self.acc_train.std()
        print(
            "Training complete\nVal Acc:   mean = {:3.1f}%, std = {:2.2f}%\n"
            "Train Acc: mean = {:3.1f}%, std = {:2.2f}%\n"
            "Time:      mean = {:3.1f}s, std = {:2.2f}s".format(
                mean_val, std_val,
                mean_train, std_train,
                mean_time, std_time)
        )

    def train_error(self):
        return self.compute_nb_errors(self.train_input, self.train_target) / \
            self.train_input.shape[0] * 100

    def test_error(self):
        return self.compute_nb_errors(self.test_input, self.test_target) / \
            self.test_input.shape[0] * 100

    def compute_nb_errors(self, input, target):
        nb_errors = 0

        for b in range(0, input.size(0), self.mini_batch_size):
            output = self(input.narrow(0, b, self.mini_batch_size))
            _, predicted_targets = torch.max(output, 1)
            for k in range(self.mini_batch_size):
                if predicted_targets[k] != target[b + k]:
                    nb_errors = nb_errors + 1

        return nb_errors

    def _update_progress(self, progress, acc_val, acc_train):
        length = 20
        status = ""
        try:
            progress = float(progress)
        except TypeError:
            progress = 0
            status = "Error: progress must be numeric\r\n"

        if progress < 0:
            progress = 0
            status = "Error: progress must be >= 0\r\n"
        if progress >= 1:
            progress = 1
            status = "Finished\n"

        block = int(round(length * progress))
        text = \
            "\rPercent: [{0}] {1:3.0f}% Val Acc:{3:5.1f}%, " \
            "Train Acc:{4:5.1f}% {2}".format(
                "#" * block + "-" * (length - block),
                round(progress * 100, 2),
                status, acc_val, acc_train
            )
        sys.stdout.write(text)
        sys.stdout.flush()

    def __str__(self):
        return "{0.__class__.__name__}(epochs={0.epochs}, " \
               "mini_batch_size={0.mini_batch_size})".format(self)


class StupidNet1(Model):
    def __init__(self, nb_hidden):
        super(StupidNet1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),         # 32 x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),   # 32 x 4 x 4
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
