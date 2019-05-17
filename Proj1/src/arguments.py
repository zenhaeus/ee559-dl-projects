import os
import argparse

parser = argparse.ArgumentParser(description='Executable for Miniproject 1 - Deep Learning')

# Main arguments
parser.add_argument('--data_size',
                    type=int, default=1000,
                    help='Number of entries in dataset to train and test on (default 1000)')

parser.add_argument('--num_epochs',
                    type=int, default=25,
                    help='Number of epochs to train for (default 25)')

parser.add_argument('--batch_size',
                    type=int, default=100,
                    help='Size of minibatches for training (default 100)')

parser.add_argument('--learning_rate',
                    type=int, default = 5e-3,
                    help='Learning rate (default 5e-3)')

parser.add_argument('--train_best',
                    action='store_true', default=False,
                    help='Train the best model 10 times')

parser.add_argument('--reproduce_all',
                    action='store_true', default=False,
                    help='Reproduce optimization')

# Prologue arguments
parser.add_argument('--full',
                    action='store_true', default=False,
                    help='Use the full set, can take ages (default False)')

parser.add_argument('--tiny',
                    action='store_true', default=False,
                    help='Use a very small set for quick checks (default False)')

parser.add_argument('--seed',
                    type=int, default=0,
                    help='Random seed (default 0, < 0 is no seeding)')

parser.add_argument('--cifar',
                    action='store_true', default=False,
                    help='Use the CIFAR data-set and not MNIST (default False)')

# Define data absolute in project root directory
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(__file__, '../../data'))
parser.add_argument('--data_dir',
                    type=str, default=DEFAULT_DATA_DIR,
                    help='Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

# Timur's fix
parser.add_argument('-f', '--file',
                    help='quick hack for jupyter')

args = parser.parse_args()
