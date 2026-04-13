"""
baseline dummy model that uniformly randomly predits true/false. Also to act as a baseline
"""

import helper_loads.load_dataloader as ld
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, random_split

# FLAG: data:

data_loaders_dict = ld.get_dataloaders_task4()
trainloader = data_loaders_dict["trainloader"]
valloader = data_loaders_dict["devloader"]


# FLAG: hyperparameters and global vars:
classes = ("fake", "real")
batch_size = 4


def random_predict_batch(batch_size, class_probs):
    return torch.multinomial(class_probs, batch_size, replacement=True)


def compute_class_distribution(loader):
    counts = torch.zeros(2)
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    probs = counts / counts.sum()
    return probs


def evald(loader, loader_name, class_probs):
    total = 0
    correct = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    for images, labels in loader:
        batch_size = labels.shape[0]
        predictions = random_predict_batch(batch_size, class_probs)

        correct += (predictions == labels).sum().item()
        total += batch_size

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    overall_acc = 100 * correct / total
    print(f"Accuracy on {loader_name}: {overall_acc:.1f}% (class-prior baseline)")

    for classname in classes:
        acc = 100 * correct_pred[classname] / total_pred[classname]
        print(f"  {classname:5s} accuracy: {acc:.1f}%")


def main():
    class_probs = compute_class_distribution(trainloader)
    print("Class distribution:", class_probs.tolist())

    evald(trainloader, "train set", class_probs)

    evald(valloader, "validation set", class_probs)


if __name__ == "__main__":
    main()
