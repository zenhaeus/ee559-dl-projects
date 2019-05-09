import glob
import torch
from matplotlib import pyplot as plt

mini_batch_sizes = [25, 50, 100, 200]

few_trains = []
for i in range(4):
    one_train = []
    for fname in glob.iglob("output/AuxNet2_{:02}*accs_val*".format(i + 1)): #avg_losses accs_val accs_train
        one_train.append(torch.load(fname))
    one_train = torch.stack(one_train)
    few_trains.append(one_train)

with plt.style.context("seaborn"):
    plt.figure(figsize=(6, 4))
    for i in range(4):
        mean = few_trains[i].mean(0)
        std = few_trains[i].std(0)
        size = list(mean.size())[0]
        plt.plot(mean.numpy(), label=mini_batch_sizes[i])
        plt.fill_between(torch.arange(size).numpy(), (mean - std).numpy(), (mean + std).numpy(), alpha=0.3)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Val Acc") #"Loss" "Val Acc" "Train Acc"
    plt.ylim(50, 80)
    #plt.yscale("log")
    plt.title("Validation Accuracy") #"Loss" "Validation Accuracy" "Training Accuracy"

plt.savefig("output/accs_val.pdf", dpi=300)
plt.close()
