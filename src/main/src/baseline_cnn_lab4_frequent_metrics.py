"""
baselinec CNN from lab 4 but shows metrics more frequently
"""

# FLAG: imports:

import sys

import helper_loads.load_dataloader as ld
import helper_loads.load_hyperparameters as lh
import helper_loads.load_transformations as lt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, random_split

# FLAG: class and architecture:


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    trainloader,
    valloader,
    net,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    show_metrics_freq=250,
):
    highest_val_accuracy = 0
    highest_val_accuracy_epoch = 0
    train_progress = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # train loop
        train_labels = []
        train_predictions = []
        temp_train_preds = []
        temp_train_labels = []
        train_i = 0
        temp_train_loss_total = 0.0
        net.train()
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            train_labels.extend(labels.cpu().numpy())
            temp_train_labels.extend(labels.cpu().numpy())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            temp_train_loss_total += loss.item()
            train_i += 1
            _, predicted = torch.max(outputs.data, 1)
            temp_train_preds.extend(predicted.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
            if train_i % show_metrics_freq == 0:
                accuracy = (
                    (torch.tensor(temp_train_preds) == torch.tensor(temp_train_labels))
                    .float()
                    .mean()
                    .item()
                )
                # show metrics
                temp_train_loss_av = temp_train_loss_total / show_metrics_freq
                print(
                    "TRAIN: Loss: "
                    + str(temp_train_loss_av)
                    + " |  Accuracy: "
                    + str(accuracy)
                )
                temp_train_preds = []
                temp_train_labels = []
                temp_train_loss_total = 0.0

            # get statistics
            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)
        # val loop
        net.eval()
        val_loss = 0.0
        temp_val_loss_total = 0.0
        all_predictions = []
        all_labels = []
        val_labels = []
        val_predictions = []
        val_i = 0
        temp_val_preds = []
        temp_val_labels = []
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                val_labels.extend(labels.cpu().numpy())
                # calculate outputs by running images through the network
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                temp_val_loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # calculate metrics
                val_predictions.extend(predicted.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                temp_val_preds.extend(predicted.cpu().numpy())
                temp_val_labels.extend(labels.cpu().numpy())
                val_i += 1
                all_labels.extend(labels.cpu().numpy())
                if val_i % show_metrics_freq == 0:
                    val_loss_av = temp_val_loss_total / show_metrics_freq
                    accuracy = (
                        (torch.tensor(temp_val_preds) == torch.tensor(temp_val_labels))
                        .float()
                        .mean()
                        .item()
                    )
                    # show metrics
                    print(
                        "VAL: Loss: "
                        + str(val_loss_av)
                        + " |  Accuracy: "
                        + str(accuracy)
                    )
                    temp_val_loss_total = 0.0
                    temp_val_preds = []
                    temp_val_labels = []
        val_loss /= len(valloader)
        accuracy = (
            (torch.tensor(all_predictions) == torch.tensor(all_labels))
            .float()
            .mean()
            .item()
        )
        if accuracy > highest_val_accuracy:
            highest_val_accuracy = accuracy
            highest_val_accuracy_epoch = epoch + 1
        precision = precision_score(all_labels, all_predictions, average="macro")
        recall = recall_score(all_labels, all_predictions, average="macro")
        # show epoch metrics
        print(
            f"EPOCH METRICS: "
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} |"
            f"Precision: {precision:.4f} |"
            f"Recall: {recall:.4f}"
        )
        train_progress.append(
            [epoch + 1, train_loss, val_loss, accuracy, precision, recall]
        )
    return (
        outputs,
        net,
        train_progress,
        highest_val_accuracy,
        highest_val_accuracy_epoch,
    )


def main():
    # FLAG: hyperparameters and global vars:
    run_name = sys.argv[1]
    print("############# Starting " + run_name + " ####################")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    net = CNN().to(device)
    # get hyperparameters
    hyperparams = lh.Hyperparameters(net)
    hyperparams_dict = hyperparams.get_random_hyperparams_dict()
    hyperparams.print_hyperparameters()
    batch_size = hyperparams_dict["batch_size"]
    num_epochs = hyperparams_dict["random_epochs"]
    optimizer = hyperparams_dict["random_optimizer"]
    transformation = hyperparams_dict["random_transformation"]

    # FLAG: data:
    data_loaders_dict = ld.get_dataloaders_task4(
        batch_size=batch_size, transform=transformation
    )
    trainloader = data_loaders_dict["trainloader"]
    valloader = data_loaders_dict["devloader"]

    criterion = nn.CrossEntropyLoss()
    outputs, net, train_progress, highest_val_accuracy, highest_val_accuracy_epoch = (
        train(
            trainloader,
            valloader,
            net,
            criterion,
            optimizer,
            device,
            num_epochs=num_epochs,
        )
    )
    # make a dataframe
    df = pd.DataFrame(
        train_progress,
        columns=["Epoch", "Train Loss", "Val Loss", "Accuracy", "Precision", "Recall"],
    )
    # keep track of training using csv
    df.to_csv(
        "/home/username/MAIN/dev/CS481/final_project/202610_csci581_project_gel_han_tri/code/task4/experiments/outputs/"
        + run_name
        + ".csv",
        index=False,
    )
    # print run results
    print(
        "Highest val accuracy: ",
        highest_val_accuracy,
        "on epoch: ",
        highest_val_accuracy_epoch,
    )
    print("############# " + run_name + " done ####################")
    # torch.save(net.state_dict(), "baseline_cnn_lab4_frequent_metrics.pth")


if __name__ == "__main__":
    main()
