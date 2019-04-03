import torch

from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch import nn

from dlc_practical_prologue import *
from arguments import args

class Model(torch.nn.Module):
    """Class to encapsulate the creation, training and performance evaluation of the Model
    """

    def __init__(self, data_size=args.data_size, epochs=args.num_epochs, mini_batch_size=args.mini_batch_size):
        """ Initialize model object
        """
        super(Model, self).__init__()
        data = generate_pair_sets(args.data_size)
        self.train_input = data[0]
        self.train_target = data[1]
        self.train_classes = data[2]
        self.test_input = data[3]
        self.test_target = data[4]
        self.test_classes = data[5]
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def pre_process(self):
        pass

    def post_process(self):
        pass

    def forward(self):
        pass

    def train(self):
        for e in range(1, self.epochs + 1):
            total_loss = 0
            for b in range(0, self.train_input.size(0), self.mini_batch_size):
                self.zero_grad()

                output = self(self.train_input.narrow(0, b, self.mini_batch_size))
                target = self.train_target.narrow(0, b, self.mini_batch_size)
                loss = self.criterion(output, target)
                loss.backward()
                total_loss += loss

                self.optimizer.step()
            print("Epoch: {}\tLoss: {}".format(e, total_loss))


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

    def __str__(self):
        return "{0.__class__.__name__}(epochs={0.epochs}, mini_batch_size={0.mini_batch_size})".format(self)

class SimpleModel(Model):
    def __init__(self):
        super(SimpleModel, self).__init__(epochs=100)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(50, 2)
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = x.view(-1, 50)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)
        return x

class BilinearModel(Model):
    def __init__(self):
        super(BilinearModel, self).__init__()
        self.conv = nn.Conv2d(2, 4, kernel_size=2, stride=1)
        self.bl = nn.Bilinear(36, 36, 2)
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        x = F.relu(F.avg_pool3d(self.conv(x), 2))
        x1 = x[:,0,:,:].view(100, 36)
        x2 = x[:,1,:,:].view(100, 36)
        x = F.relu(self.bl(x1, x2))
        return x
bm = BilinearModel()
