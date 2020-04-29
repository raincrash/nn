import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision
from torchvision import transforms

import _pickle as pickle

"""
    CIFAR10
"""


class CIFAR10:
    def __init__():
        pass

    def download_CIFAR10(download, train_transform, test_transform):
        # datasets CIFAR10
        trainset = torchvision.datasets.CIFAR10(
            root="./datasets", train=True, download=download, transform=train_transform
        )

        testset = torchvision.datasets.CIFAR10(
            root="./datasets", train=False, download=download, transform=test_transform
        )

        print("Downloaded Dataset.")

        return (trainset, testset)

    def get_CIFAR10_dataset(download=True, batch_size=4):
        """
        Load and preprocess the CIFAR-10 dataset.
        """

        # Normalize
        cifar_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # Download
        trainset, testset = download_CIFAR10(
            download, train_transform=cifar_transform, test_transform=cifar_transform
        )

        # Dataload
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        cifar_classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        return (trainloader, testloader, cifar_classes)


"""
    FashionMNIST
"""


class FashionMNIST:

    def download_FashionMNIST(self, download, train_transform, test_transform):
        # datasets FashionMNIST
        trainset = torchvision.datasets.FashionMNIST(
            root="./datasets", train=True, download=download, transform=train_transform
        )

        testset = torchvision.datasets.FashionMNIST(
            root="./datasets", train=False, download=download, transform=test_transform
        )

        print("Downloaded Dataset.")

        return (trainset, testset)

    def get_FashionMNIST_datasets(self, download=True, batch_size=4):
        """
        Load and preprocess the FashionMNIST dataset.
        """

        # Normalize
        cifar_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Download
        trainset, testset = self.download_FashionMNIST(
            download, train_transform=cifar_transform, test_transform=cifar_transform)

        # Dataload
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        FashionMNIST_classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

        return (trainset, testset, trainloader, testloader, FashionMNIST_classes)
