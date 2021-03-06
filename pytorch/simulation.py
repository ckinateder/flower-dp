import os, sys

sys.path.insert(0, os.getcwd())
from multiprocessing import Process
from pickletools import optimize
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import client
import server


def load_cifar10_data(batch_size: int) -> Union[DataLoader, DataLoader]:
    """Load cifar10 data
    Args:
        batch_size (int): batch size to be applied to the dataloaders
    Returns:
        Union[DataLoader, DataLoader]: training dataloader, testing dataloader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(
        os.path.join("data"), train=True, download=True, transform=transform
    )
    testset = CIFAR10(
        os.path.join("data"), train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader


if __name__ == "__main__":
    # privacy guarantees for (epsilon, delta)-DP
    epsilon = 0.8  # lower is better
    delta = 1 / 2e5
    l2_norm_clip = 1.5  # max euclidian norm of the weight gradients

    # cuda device if available
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # client variables
    epochs = 1  # how many epochs to go through
    batch_size = 256  # batch size for training
    min_dataset_size = 1e5  # minimum training set size

    # server variables
    num_rounds = 3  # number of train/val rounds to go through
    min_available_clients = 3  # minimum number of clients to train/val - `N``
    clients_per_round = 3  # number of clients to be selected for each round - `K`
    # K <= N
    host = "[::]:8080"

    # load data
    trainloader, testloader = load_cifar10_data(batch_size)

    # create net, optimizer, loss
    net = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)
    loss = torch.nn.CrossEntropyLoss()

    # create server process
    server_process = Process(
        target=server.main,
        args=(
            epsilon,
            delta,
            l2_norm_clip,
            num_rounds,
            min_available_clients,
            clients_per_round,
            min_dataset_size,
            host,
        ),
    )
    server_process.start()

    # create client processes
    client_processes: List[Process] = []
    for c in range(min_available_clients):
        client_processes.append(
            Process(
                target=client.main,
                args=(
                    trainloader,
                    testloader,
                    net,
                    loss,
                    epsilon,
                    delta,
                    l2_norm_clip,
                    num_rounds,
                    min_dataset_size,
                    epochs,
                    host,
                    DEVICE,
                ),
            )
        )

    # start client processes
    for c in client_processes:
        c.start()

    # finally join
    for c in client_processes:
        c.join()
    server_process.join()
