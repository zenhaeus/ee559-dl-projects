import sys
import abc
import time
import json

import torch
from torch import nn
from torch import optim

from dlc_practical_prologue import *
from arguments import args


class Model(torch.nn.Module, metaclass=abc.ABCMeta):
    """Encapsulates creation, training and evaluation of a model."""

    def __init__(
        self, data_size=args.data_size, epochs=args.num_epochs,
        batch_size=args.batch_size, lr=args.learning_rate
    ):
        """Initialize a model object."""
        super(Model, self).__init__()

        self.data_size = data_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        self.n_super = 0
        self.super_name = "super_name"

    def set_super(self, super_name):
        self.n_super = 0
        self.super_name = super_name

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

    @abc.abstractmethod
    def initialize(self):
        """Needs to be implemented in a child class."""
        return

    @abc.abstractmethod
    def forward(self):
        """Needs to be implemented in a child class."""
        return

    @abc.abstractmethod
    def get_main_out(self):
        """Needs to be implemented in a child class."""
        return

    @abc.abstractmethod
    def get_loss(self):
        """Needs to be implemented in a child class."""
        return

    def train_model(self, i):
        time_start = time.time()

        # Initialise new random data and reset weights
        self.data()
        self.apply(self.weight_reset)

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        avg_losses = []
        accs_val = []
        accs_train = []

        best = {"acc_val": 0, "acc_train": 0, "loss": 10}

        for e in range(1, self.epochs + 1):
            self.train()
            sum_loss = 0
            for b in range(0, self.train_input.size(0), self.batch_size):

                loss = self.get_loss(b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()

            self.eval()
            avg_loss = sum_loss / (self.train_input.size(0) /
                                   self.batch_size)
            cur_acc_val = 100 - self.test_error()
            cur_acc_train = 100 - self.train_error()

            best["acc_val"] = max(best["acc_val"], cur_acc_val)
            best["acc_train"] = max(best["acc_train"], cur_acc_train)
            best["loss"] = min(best["loss"], avg_loss)

            avg_losses.append(avg_loss)
            accs_val.append(cur_acc_val)
            accs_train.append(cur_acc_train)

            self._update_progress(e / self.epochs, cur_acc_val, cur_acc_train,
                                  avg_loss)

        time_end = time.time()
        time_tot = time_end - time_start

        return avg_losses, accs_val, accs_train, best, time_tot

    def train_multiple(self, n=5, save=True):
        print("Train {} times".format(n))
        print("Model: {}".format(self))
        print("N params: {}".format(self.count_params()))

        mean = {"val": 0, "train": 0}
        std = {"val": 0, "train": 0}
        bests = {"acc_val": [], "acc_train": [], "loss": []}
        times = []

        for i in range(n):
            # Train the current model
            avg_losses, accs_val, accs_train, best, time_tot = \
                self.train_model(i)

            for key in bests.keys():
                bests[key].append(best[key])
            times.append(time_tot)

            if save:
                # Save logs for the current model
                torch.save(
                    torch.tensor(avg_losses),
                    "../output/{}/{}/{}_{:02}_{:02}_avg_losses.pt".format(
                        self.super_name, self, self, self.n_super, i))

                torch.save(
                    torch.tensor(accs_val),
                    "../output/{}/{}/{}_{:02}_{:02}_accs_val.pt".format(
                        self.super_name, self, self, self.n_super, i))

                torch.save(
                    torch.tensor(accs_train),
                    "../output/{}/{}/{}_{:02}_{:02}_accs_train.pt".format(
                        self.super_name, self, self, self.n_super, i))

        if save:
            with open(
                "../output/{}/{}/{}_{:02}_bests.json".format(
                    self.super_name, self, self, self.n_super), "w"
            ) as f:
                json.dump(bests, f)

        acc_val = torch.tensor(bests["acc_val"])
        acc_train = torch.tensor(bests["acc_train"])
        time_tot = torch.tensor(times)

        mean["val"] = acc_val.mean().item()
        std["val"] = acc_val.std().item()
        mean["train"] = acc_train.mean().item()
        std["train"] = acc_train.std().item()
        mean_time, std_time = time_tot.mean(), time_tot.std()

        self.n_super += 1

        print(
            "Training complete\nBest Val Acc:   mean = {:3.1f}%, std = {:2.2f}%\n"
            "Best Train Acc: mean = {:3.1f}%, std = {:2.2f}%\n"
            "Time:           mean = {:3.1f}s, std = {:2.2f}s".format(
                mean["val"], std["val"],
                mean["train"], std["train"],
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

        for b in range(0, input.size(0), self.batch_size):
            out = self.get_main_out(b, input)
            _, predicted_targets = torch.max(out, 1)
            for k in range(self.batch_size):
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
        return "{0.__class__.__name__}".format(self)
        # (epochs={0.epochs}, batch_size={0.batch_size})


class BasicNet1(Model):
    def __init__(
        self,
        n_chns_1=16, n_chns_2=32, n_chns_3=64,
        n_hid=128
    ):
        super(BasicNet1, self).__init__()
        self.n_chns_3 = n_chns_3
        self.initialize(n_chns_1, n_chns_2, n_chns_3, n_hid)

    def initialize(self, n_chns_1=32, n_chns_2=64, n_chns_3=128, n_hid=64):
        self.n_chns_3 = n_chns_3
        self.features = nn.Sequential(
            # CNN for feature extraction
            nn.Conv2d(2, n_chns_1, kernel_size=3),                   # N x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # N x 6 x 6

            nn.Conv2d(n_chns_1, n_chns_2, kernel_size=3, padding=1),   # M x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # M x 3 x 3

            nn.Conv2d(n_chns_2, n_chns_3, kernel_size=2, padding=1),   # K x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                     # K x 1 x 1
        )

        # FC layers for binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.n_chns_3, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.n_chns_3)
        x = self.classifier(x)
        return x

    def get_main_out(self, b, input):
        return self(input.narrow(0, b, self.batch_size))

    def get_loss(self, b):
        out = self(self.train_input.narrow(0, b, self.batch_size))
        loss = self.criterion(out, self.train_target.narrow(0, b, self.batch_size))
        return loss


class AuxNet1(Model):
    def __init__(
        self,
        n_chns_1=16, n_chns_2=32, n_chns_3=64,
        n_hid=128
    ):
        super(AuxNet1, self).__init__()
        self.n_chns_3 = n_chns_3
        self.initialize(n_chns_1, n_chns_2, n_chns_3, n_hid)

    def initialize(self, n_chns_1=32, n_chns_2=64, n_chns_3=128, n_hid=64):
        self.n_chns_3 = n_chns_3
        # CNN for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, n_chns_1, kernel_size=3),                   # N x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # N x 6 x 6

            nn.Conv2d(n_chns_1, n_chns_2, kernel_size=3, padding=1),   # M x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # M x 3 x 3

            nn.Conv2d(n_chns_2, n_chns_3, kernel_size=2, padding=1),   # K x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                     # K x 1 x 1
        )

        # FC layers for digit classification
        self.classifierNumber = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_chns_3, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 10),
        )

        # FC layers for binary classification
        self.classifierFinal = nn.Sequential(
            nn.Dropout(),
            nn.Linear(20, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 2),
        )

    def forward(self, x):
        # classification of digit of channel 0
        x_num1 = self.features(x[:, :1, :, :])
        x_num1 = x_num1.view(-1, self.n_chns_3)
        x_num1 = self.classifierNumber(x_num1)

        # classification of digit of channel 1
        x_num2 = self.features(x[:, 1:, :, :])
        x_num2 = x_num2.view(-1, self.n_chns_3)
        x_num2 = self.classifierNumber(x_num2)

        # final classification
        x_f = torch.cat((x_num1, x_num2), 1)
        x_f = self.classifierFinal(x_f)

        return x_num1, x_num2, x_f

    def get_main_out(self, b, input):
        _, _, out_f = self(input.narrow(
            0, b, self.batch_size))
        return out_f

    def get_loss(self, b):
        out_1, out_2, out_f = self(self.train_input.narrow(
            0, b, self.batch_size))
        target = self.train_target.narrow(0, b, self.batch_size)
        classes = self.train_classes.narrow(0, b, self.batch_size)

        loss_1 = self.criterion(out_1, classes[:, 0])
        loss_2 = self.criterion(out_2, classes[:, 1])
        loss_f = self.criterion(out_f, target)
        loss = loss_1 + loss_2 + loss_f
        return loss


class AuxNet2(Model):
    def __init__(
            self,
            n_chns_1=16, n_chns_2=32, n_chns_3=64,
            n_hid=128
    ):
        super(AuxNet2, self).__init__()
        self.n_chns_3 = n_chns_3
        self.initialize(n_chns_1, n_chns_2, n_chns_3, n_hid)

    def initialize(self, n_chns_1=16, n_chns_2=32, n_chns_3=64, n_hid=128):
        self.n_chns_3 = n_chns_3
        # CNN for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, n_chns_1, kernel_size=3),                   # N x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # N x 6 x 6

            nn.Conv2d(n_chns_1, n_chns_2, kernel_size=3, padding=1),   # M x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # M x 3 x 3

            nn.Conv2d(n_chns_2, n_chns_3, kernel_size=2, padding=1),   # K x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                     # K x 1 x 1
        )

        # FC layers for digit classification
        self.classifier_number = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_chns_3, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 10),
        )

        # FC layers for binary classification
        self.classifier_final = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_chns_3 * 2, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 2),
        )

    def forward(self, x):
        # Channel 0 digit feature extraction
        x_1 = self.features(x[:, :1, :, :])
        x_1 = x_1.view(-1, self.n_chns_3)

        # Channel 1 digit feature extraction
        x_2 = self.features(x[:, 1:, :, :])
        x_2 = x_2.view(-1, self.n_chns_3)

        x_f = torch.cat((x_1, x_2), 1)

        out_1 = self.classifier_number(x_1)
        out_2 = self.classifier_number(x_2)
        out_f = self.classifier_final(x_f)

        return out_1, out_2, out_f

    def get_main_out(self, b, input):
        _, _, out_f = self(input.narrow(
            0, b, self.batch_size))
        return out_f

    def get_loss(self, b):
        out_1, out_2, out_f = self(self.train_input.narrow(
            0, b, self.batch_size))
        target = self.train_target.narrow(0, b, self.batch_size)
        classes = self.train_classes.narrow(0, b, self.batch_size)

        loss_1 = self.criterion(out_1, classes[:, 0])
        loss_2 = self.criterion(out_2, classes[:, 1])
        loss_f = self.criterion(out_f, target)
        loss = loss_1 + loss_2 + loss_f
        return loss


class AuxNet3(Model):
    def __init__(
            self,
            n_chns_1=16, n_chns_2=32, n_chns_3=64, n_chns_4=64,
            n_hid=64
    ):
        super(AuxNet3, self).__init__()
        self.n_chns_3 = n_chns_3
        self.n_chns_4 = n_chns_4
        self.initialize(n_chns_1, n_chns_2, n_chns_3, n_hid)

    def initialize(self, n_chns_1=16, n_chns_2=32, n_chns_3=64, n_chns_4=64, n_hid=128):
        self.n_chns_3 = n_chns_3
        self.n_chns_4 = n_chns_4
        # CNN for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, n_chns_1, kernel_size=3),                   # N x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # N x 6 x 6

            nn.Conv2d(n_chns_1, n_chns_2, kernel_size=3, padding=1),   # M x 6 x 6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                     # M x 3 x 3

            nn.Conv2d(n_chns_2, n_chns_3, kernel_size=2, padding=1),   # K x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                     # K x 1 x 1
        )

        # FC layers for digit classification
        self.classifier_number = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_chns_3, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 10),
        )

        self.features_final = nn.Sequential(
            nn.Conv2d(n_chns_3, n_chns_4, kernel_size=2, padding=1),  # 128 x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                    # 128 x 1 x 1
        )

        # FC layers for final decision
        self.classifier_final = nn.Sequential(
            nn.Dropout(),
            nn.Linear(n_chns_4 * 2, n_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hid, 2),
        )

    def forward(self, x):
        # Channel 0 digit feature extraction
        x_1 = self.features(x[:, :1, :, :])
        x_1_f = self.features_final(x_1)
        x_1 = x_1.view(-1, self.n_chns_3)
        x_1_f = x_1_f.view(-1, self.n_chns_4)

        # Channel 1 digit feature extraction
        x_2 = self.features(x[:, 1:, :, :])
        x_2_f = self.features_final(x_2)
        x_2 = x_2.view(-1, self.n_chns_3)
        x_2_f = x_2_f.view(-1, self.n_chns_4)

        x_f = torch.cat((x_1_f, x_2_f), 1)

        out_1 = self.classifier_number(x_1)
        out_2 = self.classifier_number(x_2)
        out_f = self.classifier_final(x_f)

        return out_1, out_2, out_f

    def get_main_out(self, b, input):
        _, _, out_f = self(input.narrow(
            0, b, self.batch_size))
        return out_f

    def get_loss(self, b):
        out_1, out_2, out_f = self(self.train_input.narrow(
            0, b, self.batch_size))
        target = self.train_target.narrow(0, b, self.batch_size)
        classes = self.train_classes.narrow(0, b, self.batch_size)

        loss_1 = self.criterion(out_1, classes[:, 0])
        loss_2 = self.criterion(out_2, classes[:, 1])
        loss_f = self.criterion(out_f, target)
        loss = loss_1 + loss_2 + loss_f
        return loss
