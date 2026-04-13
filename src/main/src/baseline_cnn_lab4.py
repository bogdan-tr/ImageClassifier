"""
Basic CNN from lab4 slightly adjusted to work with new data (adjusted dims and input data). serves as a baseline
"""

# FLAG: imports:

import helper_loads.load_dataloader as ld
import helper_loads.load_transformations as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader, random_split

# from torchvision.transforms import functional as TF

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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(trainloader, valloader, net, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0
        total = 0
        # train loop
        net.train()
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get statistics
            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)
        # val loop
        net.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)

                # images = gaussian_blur(images)  # TEST:
                # calculate outputs by running images through the network
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(valloader)
        accuracy = (
            (torch.tensor(all_predictions) == torch.tensor(all_labels))
            .float()
            .mean()
            .item()
        )
        precision = precision_score(all_labels, all_predictions, average="macro")
        recall = recall_score(all_labels, all_predictions, average="macro")
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} |"
            f"Precision: {precision:.4f} |"
            f"Recall: {recall:.4f}"
        )
    return outputs


def main():
    # FLAG: hyperparameters and global vars:
    batch_size = 4
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # FLAG: data:
    data_loaders_dict = ld.get_dataloaders_task4(batch_size=batch_size)
    trainloader = data_loaders_dict["trainloader"]
    valloader = data_loaders_dict["devloader"]

    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    outputs = train(
        trainloader, valloader, net, criterion, optimizer, device, num_epochs=num_epochs
    )


if __name__ == "__main__":
    main()
