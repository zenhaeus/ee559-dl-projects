#!/usr/bin/env python
import model as m
from plot import *
from get_params import *
from arguments import args


def main():

    if args.reproduce_all:
        names = ["BasicNet1", "AuxNet1", "AuxNet2", "AuxNet3"]

        for name in names:

            # Get network parameters
            get_net_params(name=name)

            # Optimize network parameters
            with open("../output/net_params/{}/params_dict.json".format(name), "r") as f:
                params_dict = json.load(f)
            print("Params loaded")

            model = getattr(m, name)
            model.set_super("net_params")
            for i in range(len(params_dict)):
                model.initialize(*params_dict[str(i)])
                print("\nParams: {}, {}, {}, {}".format(*params_dict[str(i)]))
                model.train_multiple()

        # Optimize learning rates
        with open("../output/learning_rates/AuxNet2/params_dict.json", "r") as f:
            params_dict = json.load(f)
        print("Params loaded")
        learning_rates = params_dict["learning_rates"]

        model = m.AuxNet2(32, 32, 64, 256)
        model.set_super("learning_rates")
        for lr in learning_rates:
            model.lr = lr
            print("Learning rate: {}".format(lr))
            model.train_multiple()

        # Optimize batch sizes
        with open("../output/batch_sizes/AuxNet2/params_dict.json", "r") as f:
            params_dict = json.load(f)
        print("Params loaded")
        batch_sizes = params_dict["batch_sizes"]

        model = m.AuxNet2(32, 32, 64, 256)
        model.set_super("batch_sizes")
        for batch_size in batch_sizes:
            model.batch_size = batch_size
            print("Batch size: {}".format(batch_size))
            model.train_multiple()

        # Plot validation and training training curves
        plot_loss_accs(path="learning_rates", type_="accs_val")
        plot_loss_accs(path="learning_rates", type_="accs_train")
        plot_loss_accs(path="batch_sizes", type_="accs_val")
        plot_loss_accs(path="batch_sizes", type_="accs_train")

        # Plot mean and std of best accuracies for each
        # varied parameter
        # Network parameters
        plot_best_net_params("BasicNet1")
        plot_best_net_params("AuxNet1")
        plot_best_net_params("AuxNet2")
        plot_best_net_params("AuxNet3")

        # Learning rates and batch sizes
        plot_best_lr_batch("learning_rates")
        plot_best_lr_batch("batch_sizes")

    if args.train_best:
        # Train the best model 10 times
        model = m.AuxNet2(32, 32, 64, 256)
        print("Learning rate: {}".format(model.lr))
        print("Batch size: {}".format(model.batch_size))
        model.train_multiple(n=10, save=False)

    else:
        # Train the fastest and good model 10 times
        print("Train the fastest sufficient model")
        model = m.AuxNet3(16, 32, 64, 128)
        model.batch_size = 200
        print("Learning rate: {}".format(model.lr))
        print("Batch size: {}".format(model.batch_size))
        model.train_multiple(n=10, save=False, evaluate=False)

    print("End of main()")


if __name__ == '__main__':
    main()
