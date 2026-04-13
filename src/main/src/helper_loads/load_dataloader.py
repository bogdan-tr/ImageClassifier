"""
conviniently load data and return as dataloader object that's easy to adjust
"""

import torch
import torchvision.transforms as transforms
from helper_loads import load_transformations as lt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets import ImageFolder

transformations = lt.transformations()
default_transform = transformations.basic_transform
val_transform = transformations.val_transform


class fake_real_image_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def get_dataloader_generic(data_dir, batch_size, num_workers, shuffle, transform):
    """returns dataloader object for given data directory (generic)"""
    # get dataset object
    dataset = fake_real_image_dataset(data_dir, transform=transform)
    # datasets.ImageFolder(root=data_dir, transform=transform)

    # wrapt into dataloader object and return
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    return dataloader


def get_dataloaders_task4(
    batch_size=4,
    num_workers=4,
    shuffle=True,
    transform=default_transform,
    val_transform=val_transform,
):
    """returns dictionary with dataloader objects for task4 (project specifc), (shuffle arg
    applies only to trainig set, devset shuffle fixed to false)"""

    trainset = get_dataloader_generic(
        "/local/202610_csci581_project/project_data/task4/train/",
        batch_size,
        num_workers,
        shuffle,
        transform,
    )

    devset = get_dataloader_generic(
        "/local/202610_csci581_project/project_data/task4/dev/",
        batch_size,
        num_workers,
        False,
        val_transform,
    )

    return {
        "trainloader": trainset,
        "devloader": devset,
    }
