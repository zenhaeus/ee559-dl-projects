import torch

class Model(torch.nn.Sequential):
    """Class to encapsulate the creation, training and performance evaluation of the Model
    """

    def __init__(self, *layers, data):
        """ Initialize model object 
        """
        super(Model, self).__init__(layers)
        self.train_input = data[0]
        self.train_target = data[1]
        self.train_classes = data[2]
        self.test_input = data[3]
        self.test_target = data[4]
        self.test_classes = data[5]

    def train(self):
        """ Train the model with the train data
        """

        pass

    def get_performance_metrics():
        pass
