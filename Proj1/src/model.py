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

    def weight_reset(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.reset_parameters()

    def count_params(self):
        ps = filter(lambda p: p.requires_grad, self.parameters())
        return sum([torch.tensor(p.size()).prod().item() for p in ps])

    def set_criterion(self, criterion):
        self.criterion = criterion

    @abc.abstractmethod
    def forward(self):
        """Needs to be implemented in a child class."""
        return

    def train(self, i):
        time_start = time.time()

        # Initialise new random data and reset weights
        self.data()
        self.apply(self.weight_reset)

        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        avg_losses = []
        accs_val = []
        accs_train = []

        for e in range(1, self.epochs + 1):
            sum_loss = 0
            for b in range(0, self.train_input.size(0), self.mini_batch_size):

                loss = self.get_loss(b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()

            avg_loss = sum_loss / (self.train_input.size(0) / self.mini_batch_size)
            cur_acc_val = 100 - self.test_error()
            cur_acc_train = 100 - self.train_error()
            self._update_progress(e / self.epochs, cur_acc_val, cur_acc_train, avg_loss)

            avg_losses.append(avg_loss)
            accs_val.append(cur_acc_val)
            accs_train.append(cur_acc_train)
        """
        torch.save(
            torch.tensor(avg_losses),
            "output/AuxNet2_04_{:02}_avg_losses_200.pt".format(i))

        torch.save(
            torch.tensor(accs_val),
            "output/AuxNet2_04_{:02}_accs_val_200.pt".format(i))

        torch.save(
            torch.tensor(accs_train),
            "output/AuxNet2_04_{:02}_accs_train_200.pt".format(i))
        """

        self.acc_train.append(cur_acc_train)
        self.acc_val.append(cur_acc_val)

        time_end = time.time()
        self.time.append(time_end - time_start)

    def train_multiple(self, n=3):
        print("Train {} times".format(n))
        print("Model: {}".format(self))
        print("N params: {}".format(self.count_params()))
        self.acc_train, self.acc_val = [], []

        for i in range(n):
            self.train(i)

        acc_val = torch.tensor(self.acc_val)
        acc_train = torch.tensor(self.acc_train)
        time = torch.tensor(self.time)

        mean_val, std_val = acc_val.mean(), acc_val.std()
        mean_train, std_train = acc_train.mean(), acc_train.std()
        mean_time, std_time = time.mean(), time.std()
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
            out = self.get_main_out(b, input)
            _, predicted_targets = torch.max(out, 1)
            for k in range(self.mini_batch_size):
                if predicted_targets[k] != target[b + k]:
                    nb_errors = nb_errors + 1

        return nb_errors

    def _update_progress(self, progress, acc_val, acc_train, avg_loss):
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
            status = "Fin\n"

        block = int(round(length * progress))
        text = \
            "\rPercent: [{0}] {1:3.0f}% VAcc:{3:5.1f}%, " \
            "TAcc:{4:5.1f}% TL:{5:5.2f} {2}".format(
                "#" * block + "-" * (length - block),
                round(progress * 100, 2),
                status, acc_val, acc_train, avg_loss
            )
        sys.stdout.write(text)
        sys.stdout.flush()

    def __str__(self):
        return "{0.__class__.__name__}(epochs={0.epochs}, " \
               "mini_batch_size={0.mini_batch_size}".format(self)


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
            #nn.Dropout(),
            nn.Linear(64 * 2 * 2, nb_hidden),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(nb_hidden, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 2 * 2)
        x = self.classifier(x)
        return x

    def get_main_out(self, b, input):
        return self(input.narrow(0, b, self.mini_batch_size))

    def get_loss(self, b):
        out = self(self.train_input.narrow(0, b, self.mini_batch_size))
        loss = self.criterion(out, self.train_target.narrow(0, b, self.mini_batch_size))
        return loss


class FancyNet2(Model):
    def __init__(self):
        super(FancyNet2, self).__init__()
        self.n_chns_1 = 16
        self.n_chns_2 = 32
        self.n_chns_3 = 64
        self.n_hid = 128

        # CNN for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, self.n_chns_1, kernel_size=3),          # 32 x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),               # 32 x 6 x 6

            nn.Conv2d(self.n_chns_1, self.n_chns_2, kernel_size=3, padding=1),   # 64 x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # 64 x 3 x 3

            nn.Conv2d(self.n_chns_2, self.n_chns_3, kernel_size=2, padding=1),  # 128 x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                    # 128 x 1 x 1
        )

        # FC layers for digit classification
        self.classifier_number = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.n_chns_3, self.n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.n_hid, 10),
        )

        # FC layers for final decision
        self.classifier_final = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.n_chns_3 * 2, self.n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.n_hid, 2),
        )

    def forward(self, x):
        # Channel 0 digit feature extractin
        x_1 = self.features(x[:, :1, :, :])
        x_1 = x_1.view(-1, self.n_chns_3)

        # Channel 1 digit feature extractin
        x_2 = self.features(x[:, :1, :, :])
        x_2 = x_2.view(-1, self.n_chns_3)

        x_f = torch.cat((x_1, x_2), 1)

        out_1 = self.classifier_number(x_1)
        out_2 = self.classifier_number(x_2)
        out_f = self.classifier_final(x_f)

        return out_1, out_2, out_f

    def get_main_out(self, b, input):
        _, _, out_f = self(input.narrow(
            0, b, self.mini_batch_size))
        return out_f

    def get_loss(self, b):
        out_1, out_2, out_f = self(self.train_input.narrow(
            0, b, self.mini_batch_size))
        target = self.train_target.narrow(0, b, self.mini_batch_size)
        classes = self.train_classes.narrow(0, b, self.mini_batch_size)

        loss_1 = self.criterion(out_1, classes[:, 0])
        loss_2 = self.criterion(out_2, classes[:, 1])
        loss_f = self.criterion(out_f, target)
        loss = loss_1 + loss_2 + loss_f
        return loss
