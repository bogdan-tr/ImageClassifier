import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""importable script for quicly analyzing and generating massive amounts of plots from csv files generate during
random search"""


def plot_losses_single(df, filename):
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", color="orange")
    plt.plot(df["Epoch"], df["Val Loss"], label="Validation Loss", color="yellow")
    plt.legend(loc="upper left")
    plt.ylabel("Loss")
    plt.twinx()
    plt.plot(
        df["Epoch"], df["Accuracy"], label="Accuracy", color="green", linestyle="--"
    )
    plt.legend(loc="upper right")
    plt.title("Train vs Dev Loss and Acc. for " + filename)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("plots/" + filename + ".png")
    # plt.show()


def plot_losses_double(df1, df2, filename1, filename2):
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.plot(
        df1["Epoch"], df1["Train Loss"], label="train_loss " + filename1, color="orange"
    )
    plt.plot(
        df1["Epoch"], df1["Val Loss"], label="dev_loss " + filename1, color="yellow"
    )
    plt.plot(
        df2["Epoch"],
        df2["Train Loss"],
        label="train_loss " + filename2,
        color="royalblue",
    )
    plt.plot(
        df2["Epoch"],
        df2["Val Loss"],
        label="dev_loss " + filename2,
        color="mediumpurple",
    )
    plt.legend(loc="upper left")
    plt.ylabel("Loss")
    plt.twinx()
    plt.plot(
        df1["Epoch"],
        df1["Accuracy"],
        label="accuracy " + filename1,
        color="green",
        linestyle="--",
    )
    plt.plot(
        df2["Epoch"],
        df2["Accuracy"],
        label="accuracy " + filename2,
        color="springgreen",
        linestyle="--",
    )
    plt.legend(loc="upper right")
    plt.title("Train vs Dev Loss and Acc. for " + filename1 + " vs " + filename2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("plots/" + filename1 + "_vs_" + filename2 + ".png")
    # plt.show()


def plot_precision_recall_double(df1, df2, filename1, filename2):
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.plot(
        df1["Epoch"], df1["Precision"], label="precision " + filename1, color="orange"
    )
    plt.plot(df1["Epoch"], df1["Recall"], label="recall " + filename1, color="yellow")
    plt.plot(
        df2["Epoch"],
        df2["Precision"],
        label="precision " + filename2,
        color="royalblue",
    )
    plt.plot(
        df2["Epoch"],
        df2["Recall"],
        label="recall " + filename2,
        color="mediumpurple",
    )
    plt.legend(loc="upper left")
    plt.twinx()
    plt.plot(
        df1["Epoch"],
        df1["Accuracy"],
        label="accuracy " + filename1,
        color="green",
        linestyle="--",
    )
    plt.plot(
        df2["Epoch"],
        df2["Accuracy"],
        label="accuracy " + filename2,
        color="springgreen",
        linestyle="--",
    )
    plt.legend(loc="upper right")
    plt.title("Precision, Recall for " + filename1 + " vs " + filename2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("plots/" + filename1 + "_vs_" + filename2 + "_p_vs_r" + ".png")
    # plt.show()


def plot_all_accuracy(dir):
    base_dir = "/home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/"
    file_acc_dict = {}
    for filename in os.listdir(os.path.join(base_dir, dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(base_dir, dir, filename))
            max_acc = df["Accuracy"].max()
            filename.replace(".csv", "")
            file_acc_dict[filename] = max_acc
    k = list(file_acc_dict.keys())
    v = list(file_acc_dict.values())
    idx = np.argsort(v)
    sorted_dict = {k[i]: v[i] for i in idx}
    filenames = list(sorted_dict.keys())
    max_accs = list(sorted_dict.values())
    plt.style.use("dark_background")
    plt.figure(figsize=(20, 6))  # CONT:
    plt.bar(filenames, max_accs, color="green")
    plt.xlabel("Run Name")
    plt.ylabel("Highest Accuracy")
    plt.title("Highest Accuracy per Run")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.5, 1)
    plt.tight_layout()
    dir = dir.replace("/", "")
    dir = dir.replace(".", "")
    plt.savefig("plots/" + dir + "_max_accuracy.png")
    print()  # CONT: print stats
    # plt.show()


def generate_all_plots(dir):
    plot_all_accuracy(dir)
    base_dir = "/home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/"
    for filename in os.listdir(os.path.join(base_dir, dir)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(base_dir, dir, filename))
            filename.replace(".csv", "")
            plot_losses_single(df, filename)


def main():
    # modes for running
    dir = sys.argv[1]
    generate_all_plots(dir)
    # single_file = False
    # dir = True
    # if single_file:
    #     filename1 = sys.argv[1]
    #     df1 = pd.read_csv("../outputs/" + filename1 + ".csv")
    #     plot_losses_single(df1, filename1)
    # if not single_file and not dir:
    #     filename1 = sys.argv[1]
    #     filename2 = sys.argv[2]
    #     df1 = pd.read_csv("../outputs/" + filename1 + ".csv")
    #     df2 = pd.read_csv("../outputs/" + filename2 + ".csv")
    #     plot_losses_double(df1, df2, filename1, filename2)
    #     plot_precision_recall_double(df1, df2, filename1, filename2)
    # if dir:
    #     directory = sys.argv[1]
    #     plot_all_accuracy(directory)


if __name__ == "__main__":
    main()
