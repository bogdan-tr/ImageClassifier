import random
import sys

import torch

from helper_loads import load_transformations as lt

"""this script lets you easily load hyperparameters for random search as every instanc of the Hyperparameters class
generate random hyperparams within a certain range. """

transformations = lt.transformations()
transform_dict = {
    0: transformations.basic_transform,
    1: transformations.flip_blur_cj,
    2: transformations.random_erasing,
    3: transformations.jpeg_compression,
    4: transformations.all_transforms,
}

transform_name_dict = {
    0: "basic_transform",
    1: "flip_blur_cj",
    2: "random_erasing",
    3: "jpeg_compression",
    4: "all_transforms",
}

optimizer_dict = {
    0: "adam",
    1: "adamw",
    # 2: "adagrad",
    # 3: "adadelta",
    2: "rmsprop",
    # 5: "sgd",
}


class Hyperparameters:
    def __init__(self, model):
        self.random_transformation_id = random.randint(0, 4)
        self.random_transformation = self.get_random_transformation(
            self.random_transformation_id
        )
        self.model = model
        self.random_lr_id = random.uniform(0.0008, 0.0012)
        self.random_lr = self.get_random_lr(self.random_lr_id)
        self.random_optimizer_id = random.randint(0, 2)
        self.random_optimizer = self.get_optimizer(
            self.model, self.random_lr, self.random_optimizer_id
        )
        self.random_epochs = random.randint(40, 45)
        self.batch_size = 32

    def get_random_hyperparams_dict(self):
        return {
            "random_transformation": self.random_transformation,
            "random_lr": self.random_lr,
            "random_optimizer": self.random_optimizer,
            "random_epochs": self.random_epochs,
            "batch_size": self.batch_size,
        }

    def print_hyperparameters(self):
        print("----------HYPERPARAMETERS----------")
        print(":::::OVERVIEW:::::")
        print("Transformation: ", transform_name_dict[self.random_transformation_id])
        print(f"Learning rate: {self.random_lr_id:.3f}")
        print("Optimizer: ", optimizer_dict[self.random_optimizer_id])
        print("Epochs: ", self.random_epochs)
        print("Batch size: ", self.batch_size)
        print(":::::DETAILED:::::")
        print("MODEL:", self.model)
        print("TRANSFORMATION:", self.random_transformation)
        print("LEARNING RATE:", self.random_lr)
        print("OPTIMIZER:", self.random_optimizer)
        print("EPOCHS:", self.random_epochs)
        print("BATCH SIZE:", self.batch_size)
        print("----------HYPERPARAMETERS----------")

    def get_random_transformation(self, id):
        return transform_dict[id]

    def get_random_lr(self, id):
        return id

    def get_optimizer(self, model, lr, id):
        """get the correct optimizer based on the args.opt argument"""
        match optimizer_dict[id]:
            case "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                return optimizer
            case "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                return optimizer
            case "adagrad":
                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
                return optimizer
            case "adadelta":
                optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
                return optimizer
            case "rmsprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                return optimizer
            case "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                return optimizer
            case _:
                print("OPTIMZER ARG ERROR")
                sys.exit(1)


"""
batch size (find optimal) = 40 (crashes at 48)
dropout
dataloader input transformation
num epochs
optimizer, learning rate 
"""
