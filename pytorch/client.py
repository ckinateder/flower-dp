import os
from collections import OrderedDict
from typing import Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import privacy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(batch_size: int) -> Union[DataLoader, DataLoader, int]:
    """Load CIFAR-10 (training and test set)."""
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
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        batch_size: int,
        epochs: int,
        l2_norm_clip: float,
        noise_multiplier: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        # init net
        self.net = Net().to(DEVICE)
        self.privacy_engine = PrivacyEngine(secure_mode=True)
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self) -> None:
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        self.net, optimizer, trainloader = self.privacy_engine.make_private(
            module=self.net,
            optimizer=optimizer,
            data_loader=self.trainloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.l2_norm_clip,
        )
        """
        optimizer = privacy.LaplaceDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.l2_norm_clip,
            expected_batch_size=self.batch_size,
            secure_mode=True,
        )
        """

        for _ in range(self.epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                loss = criterion(self.net(images), labels)
                loss.backward()  # compute gradients
                """
                # clip gradients and add noise
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=l2_norm_clip
                )

                with torch.no_grad():
                    for p in self.net.parameters():
                        new_val = p + np.random.laplace(
                            loc=0, scale=1 / noise_multiplier
                        )
                        p.copy_(new_val)
                """
                optimizer.step()  # apply gradients
                optimizer.zero_grad()

    def test(self) -> Union[float, float]:
        """Validate the network on the entire test set."""
        self.net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self.net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


if __name__ == "__main__":
    epochs = 1
    batch_size = 32
    l2_norm_clip = 1.5
    noise_multiplier = 0.3

    # Load model and data
    trainloader, testloader, num_examples = load_data(batch_size)

    # Start Flower client

    client = CifarClient(
        trainloader,
        testloader,
        batch_size=batch_size,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
