import abc

import torch
import mytorch.nn
import mytorch.data
import mytorch.optim



class Trainer(metaclass=abc.ABCMeta):
    """ Class to facilitate the training and evaluation of models """

    def __init__(self, model, data, optimizer=None, criterion=None, uniform_wi=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = 1e-2
        self.momentum = 0.9

        self.train_input = data[0]
        self.train_target = data[1]
        self.test_input = data[2]
        self.test_target = data[3]
        self.nb_epochs = None
        self.mini_batch_size = None
        if uniform_wi:
            self.weight_initialization()


    def train(self, mini_batch_size):
        self.mini_batch_size = 1 if mini_batch_size is None else mini_batch_size

        # Convert to one hot for training with MSE
        self.train_target_onehot = mytorch.data.target_to_onehot(self.train_target)

    def compute_nb_errors(self, data_input, data_target):
        """ Calculates the number of errors between data_input and data_target """
        if self.mini_batch_size is None:
            print("Error: Can't compute error of untrained network!")
            return -1
        else:
            return 0

    def get_train_err(self):
        nb_errors = self.compute_nb_errors(self.train_input, self.train_target)
        if nb_errors != -1:
            return  100 * nb_errors / self.train_input.size(0)
        else:
            return -1

    def get_test_err(self):
        nb_errors = self.compute_nb_errors(self.test_input, self.test_target)
        if nb_errors != -1:
            return  100 * nb_errors / self.test_input.size(0)
        else:
            return -1

    def weight_initialization(self):
        """ uniformly initialize all parameters to compare mytorch and pytorch """
        raise NotImplementedError

    def print_summary(self):
        print(
            "\n"
            "Training complete\n"
            "Train Error: {:3.1f}%\n"
            "Test Error: {:3.1f}%\n".format(
                self.get_train_err(),
                self.get_test_err()
            )
        )


class MyTorchTrainer(Trainer):
    """ Helper class to train MyTorch models. """

    def __init__(self, model, data, optimizer=None, criterion=None, uniform_wi=False):
        if not isinstance(model, mytorch.nn.Module):
            raise TypeError

        super(MyTorchTrainer, self).__init__(model, data, optimizer)
        if self.criterion is None:
            self.criterion = mytorch.nn.LossMSE()
        if self.optimizer is None:
            self.optimizer = mytorch.optim.SGD(self.model.param(), self.lr, self.momentum)

    def train(self, nb_epochs, mini_batch_size=None):
        super(MyTorchTrainer, self).train(mini_batch_size)

        print('Training of mytorch model  -------')

        for e in range(0, nb_epochs):
            sum_loss = 0

            # train in minibatches
            # TODO : if mini_batch_size != 1, results differ from PyTorch implementation
            for k in range(0, int(self.train_input.size(0) / self.mini_batch_size)):
                # set gradients to zero
                self.optimizer.zero_grad()

                for b in range (self.mini_batch_size):
                    # forward pass
                    output = self.model(self.train_input[k+b])
                    loss = self.criterion(output, self.train_target_onehot[k+b])

                    # backward pass
                    self.model.backward(self.criterion.backward())
                    sum_loss += loss.item()

                # one gradient step per minibatch
                self.optimizer.step()


            print('epoch: ', e, 'loss:', sum_loss)
    #    print("Final output:\n{}".format(model(train_input)))

    def compute_nb_errors(self, data_input, data_target):
        nb_data_errors = super(MyTorchTrainer, self).compute_nb_errors(data_input, data_target)
        if nb_data_errors == -1:
            return -1

        for k in range(data_input.size(0)):
            output = self.model(data_input[k])
            self.model.backward(output)  # TODO: this is only to free the saved input during forward pass
            predicted_class = torch.argmax(output)
            if data_target[k] != predicted_class:
                nb_data_errors = nb_data_errors + 1

        return nb_data_errors

    def weight_initialization(self):
        for p, _ in self.model.param():
            p.fill_(1e-6)


class PyTorchTrainer(Trainer):
    """ Helper class to train PyTorch models. """

    def __init__(self, model, data, optimizer=None, criterion=None, uniform_wi=False):
        if not isinstance(model, torch.nn.Module):
            raise TypeError

        super(PyTorchTrainer, self).__init__(model, data, optimizer)

    def train(self, nb_epochs, mini_batch_size=None):
        super(PyTorchTrainer, self).train(mini_batch_size)

        print('Training of pytorch model  -------')

        if self.criterion is None:
            self.criterion = torch.nn.MSELoss(reduction = "mean")
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum)

        for e in range(nb_epochs):
            sum_loss = 0

            for b in range(0, self.train_input.size(0), self.mini_batch_size):
                output = self.model(self.train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, self.train_target_onehot.narrow(0, b, self.mini_batch_size))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()

            print('epoch: ', e, 'loss:', sum_loss)

    def compute_nb_errors(self, data_input, data_target):
        nb_data_errors = super(PyTorchTrainer, self).compute_nb_errors(data_input, data_target)
        if nb_data_errors == -1:
            return -1

        for b in range(0, data_input.size(0), self.mini_batch_size):
            output = self.model(data_input.narrow(0, b, self.mini_batch_size))
            predicted_classes = torch.argmax(output, 1)
            for k in range(self.mini_batch_size):
                if data_target[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors

    def weight_initialization(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.fill_(1e-6)
