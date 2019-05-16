import glob
import torch
import json
from matplotlib import pyplot as plt


def plot_loss_accs(
    path="learning_rates",
    type_={"accs_val", "accs_train", "avg_losses"}
):
    titles = {
        "avg_losses": "Loss",
        "accs_val": "Validation Accuracy",
        "accs_train": "Training Accuracy"
    }

    with open("../output/{}/AuxNet2/params_dict.json".format(path), "r") as f:
        params_dict = json.load(f)
    print("Params loaded")
    params = params_dict[path]

    few_trains = []
    for i in range(len(params)):
        one_train = []
        for fname in glob.iglob("../output/{}/AuxNet2/AuxNet2_{:02}*{}*".format(path, i, type_)):
            one_train.append(torch.load(fname))
        one_train = torch.stack(one_train)
        few_trains.append(one_train)

    with plt.style.context("ggplot"):
        plt.figure(figsize=(6, 4))
        for i in range(len(params)):
            mean = few_trains[i].mean(0)
            std = few_trains[i].std(0)
            size = list(mean.size())[0]
            plt.plot(mean.numpy(), label=str(params[i]))
            plt.fill_between(
                torch.arange(size).numpy(),
                (mean - std).numpy(), (mean + std).numpy(),
                alpha=0.3
            )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Train Acc, %")
        #plt.ylim(50, 100)
        #plt.yscale("log")
        plt.title(titles[type_])
        plt.subplots_adjust(bottom=0.14)

    plt.savefig("../output/{}/AuxNet2/{}.pdf".format(path, type_), dpi=300)
    plt.close()


def plot_best_net_params(name="BasicNet1"):
    with open("../output/net_params/{}/params_dict.json".format(name), "r") as f:
        params_dict = json.load(f)
    print("Params loaded")

    mean_bests = {"acc_val": [], "acc_train": [], "loss": []}
    std_bests = {"acc_val": [], "acc_train": [], "loss": []}
    for i in range(len(params_dict["acc_val"])):
        fname = "../output/net_params/{}/{}_{:02}_bests.json".format(name, name, i)
        with open(fname, "r") as f:
            bests = json.load(f)
        for key in bests.keys():
            bests_torch = torch.tensor(bests[key])
            mean_bests[key].append(bests_torch.mean().item())
            std_bests[key].append(bests_torch.std().item())

    labels = {"acc_val": "Validation Accuracy", "acc_train": "Training Accuracy"}

    with plt.style.context("ggplot"):
        plt.rcParams.update({'font.size': 8})
        fig, ax = plt.subplots(figsize=(6, 4))
        mean_sorted_inds = sorted(
            range(len(mean_bests["acc_val"])),
            key=lambda k: mean_bests["acc_val"][k]
        )
        print(mean_sorted_inds)

        for key in ["acc_val", "acc_train"]:
            mean_sorted = [mean_bests[key][i] for i in mean_sorted_inds]
            std_sorted = [std_bests[key][i] for i in mean_sorted_inds]
            ax.errorbar(
                range(len(params_dict["acc_val"])), mean_sorted, std_sorted,
                alpha=0.6, capsize=2, label=labels[key]
            )
        plt.legend()

        params_values = list(params_dict.values())
        params_values_sorted = [params_values[i] for i in mean_sorted_inds]
        labels = ["{}-{}-{}-{}".format(*v) for v in params_values_sorted]

        ax.set_xticks(range(len(params_dict["acc_val"])))
        ax.set_xticklabels(labels, rotation=45, fontsize=6)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
        plt.subplots_adjust(bottom=0.17)

        plt.ylabel("Accuracy, %")
        #plt.ylim(82, 100)

        plt.title(name)

    plt.savefig("../output/net_params/{}/"
                "accs_val_train_best.pdf".format(name), dpi=300)
    plt.close()

def plot_best_lr_batch(path={"batch_sizes", "learning_rates"}, name="AuxNet2"):
    labels = {"learning_rates": "Learning Rate", "batch_sizes": "Batch Size"}

    with open("../output/{}/{}/params_dict.json".format(path, name), "r") as f:
        params_dict = json.load(f)
    print("Params loaded") 

    mean_bests = {"acc_val": [], "acc_train": [], "loss": []}
    std_bests = {"acc_val": [], "acc_train": [], "loss": []}
    for i in range(len(params_dict[path])):
        fname = "../output/{}/{}/{}_{:02}_bests.json".format(path, name, name, i)
        with open(fname, "r") as f:
            bests = json.load(f)
        for key in bests.keys():
            bests_torch = torch.tensor(bests[key])
            mean_bests[key].append(bests_torch.mean().item())
            std_bests[key].append(bests_torch.std().item())

    with plt.style.context("ggplot"):
        plt.rcParams.update({'font.size': 8})
        fig, ax = plt.subplots(figsize=(6, 4))
        for key in ["acc_val", "acc_train"]:
            ax.errorbar(
                range(len(params_dict[path])), mean_bests[key], std_bests[key],
                alpha=0.6, capsize=2, label=labels[path]
            )
        plt.legend()

        labels = params_dict[path]

        ax.set_xticks(range(len(params_dict[path])))
        ax.set_xticklabels(labels, rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
        plt.subplots_adjust(bottom=0.14)

        plt.ylabel("Accuracy, %")
        #plt.ylim(82, 100)

        plt.title(name)

    plt.savefig("../output/{}/{}/accs_val_train_best.pdf".format(path, name), dpi=300)
    plt.close()
