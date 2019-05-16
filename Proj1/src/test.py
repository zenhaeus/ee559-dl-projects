#!/usr/bin/env python
from model import *
from plot import *


def main():
    # Optimize Network Parameters
    """
    with open("../output/net_params/BasicNet1/params_dict.json", "r") as f:
        params_dict = json.load(f)
    print("Params loaded")

    model = BasicNet1()
    model.set_super("net_params")
    for i in range(len(params_dict)):
        model.initialize(*params_dict[str(i)])
        print("\nParams: {}, {}, {}, {}".format(*params_dict[str(i)]))
        model.train_multiple()
    """

    # Optimize Learning Rates
    """
    with open("../output/learning_rates/BasicNet1/params_dict.json", "r") as f:
        params_dict = json.load(f)
    print("Params loaded")
    learning_rates = params_dict["learning_rates"]

    model = AuxNet2(64, 64, 64, 64)
    model.set_super("learning_rates")
    for lr in learning_rates:
        model.lr = lr
        print("Learning rate: {}".format(lr))
        model.train_multiple()
    """

    # Optimize Batch Sizes
    """
    with open("../output/batch_sizes/BasicNet1/params_dict.json", "r") as f:
        params_dict = json.load(f)
    print("Params loaded")
    batch_sizes = params_dict["batch_sizes"]

    model = AuxNet2(64, 64, 64, 64)
    model.set_super("batch_sizes")
    for batch_size in batch_sizes:
        model.batch_size = batch_size
        print("Batch size: {}".format(batch_size))
        model.train_multiple()
    """

    """
    # Plot validation and training training curves
    plot_loss_accs(path="learning_rates", type_="accs_val")
    plot_loss_accs(path="learning_rates", type_="accs_train")
    plot_loss_accs(path="batch_sizes", type_="accs_val")
    plot_loss_accs(path="batch_sizes", type_="accs_train")
    """

    """
    # Plot mean and std of best accuracies for each
    # varied parameter
    # Network parameters
    plot_best_net_params("BasicNet1")
    plot_best_net_params("AuxNet1")
    plot_best_net_params("AuxNet2")

    # Learning rates and batch sizes
    plot_best_lr_batch("learning_rates")
    plot_best_lr_batch("batch_sizes")
    """

    # Train the best model 10 times
    model = AuxNet2(64, 64, 64, 64)
    print("Learning rate: {}".format(model.lr))
    print("Batch size: {}".format(model.batch_size))
    model.train_multiple(n=10, save=False)

    print("End of main()")


if __name__ == '__main__':
    main()
