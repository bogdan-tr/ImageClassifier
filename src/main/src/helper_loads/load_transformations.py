import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

"""This script defines a bunch of transformations useful for image fake image detection. Let's you easily
import a predefined transformation (also used in random search)"""


class transformations:
    def __init__(self):
        # basic transform from lab4
        self.basic_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # transforms inspired by "a new approach to improve learning based
        # deepfake detection in realistic condiitons paper"
        self.flip_blur_cj = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # also recommended by literature, forces model look at many facial featutres
        self.random_erasing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 0.3)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # recommmned by literature
        self.jpeg_compression = transforms.Compose(
            [
                v2.JPEG((75, 90)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.all_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
                v2.JPEG((65, 90)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=0.04, scale=(0.02, 0.15), ratio=(0.3, 0.3)),
            ]
        )
        # for validation set
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
