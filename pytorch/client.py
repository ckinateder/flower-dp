import os
from collections import OrderedDict
from typing import Optional, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from torch.nn.utils import vector_to_parameters, parameters_to_vector


import privacy

# cuda device
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
    return trainloader, testloader


class Net(nn.Module):
    """Simple CNN to be used with CIFAR10"""

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


class DPSGD(torch.optim.SGD):
    """Differentially private extension of SGD."""

    def __init__(
        self,
        noise_multiplier: float = 0.5,
        l2_norm_clip: float = 1.5,
        *args,
        **kwargs,
    ) -> None:
        """[summary]

        Args:
            noise_multiplier (float, optional): Noise multiplier. Defaults to 0.5.
            l2_norm_clip (float, optional): Max euclidian norm. Defaults to 1.5.
        """
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip

    def step(self, closure=None) -> Optional[float]:
        closure = super().step()
        params = []  # for extracted params

        # extract parameters from self.param_groups
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)

        # apply noise
        privacy.noise_and_clip_parameters(
            params,
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
        )
        return closure


"""


param_vector = parameters_to_vector(self.net.parameters())

privacy.clip_parameter(param_vector, clip_threshold=self.l2_norm_clip)
privacy.noise_parameter(
param_vector, std=self.noise_multiplier * self.l2_norm_clip
)
vector_to_parameters(param_vector, self.net.parameters())
"""


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        batch_size: int = 32,
        epochs: int = 1,
        l2_norm_clip: float = 1.5,
        noise_multiplier: float = 0.3,
        learning_rate: float = 0.001,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.net = Net().to(DEVICE)  # init net
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = {
            "trainset": len(trainloader.dataset),
            "testset": len(testloader.dataset),
            "total": len(trainloader.dataset) + len(testloader.dataset),
        }
        self.privacy_spent = None
        self.target_delta = 1 / self.num_examples["total"]
        self.learning_rate = learning_rate

    def train(self) -> None:
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()  # loss function
        optimizer = DPSGD(
            params=self.net.parameters(),
            noise_multiplier=self.noise_multiplier,
            l2_norm_clip=self.l2_norm_clip,
            lr=self.learning_rate,
            momentum=0.9,
        )
        # put in train mode
        self.net.train()

        for _ in range(self.epochs):
            for images, labels in tqdm(self.trainloader, leave=False):
                # send to device and compute loss
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                loss = criterion(self.net(images), labels)
                optimizer.zero_grad()  # zero_grad
                loss.backward()
                optimizer.step()

    def test(self) -> Union[float, float]:
        """Validate the network on the entire test set."""
        self.net.eval()  # put in test mode
        criterion = torch.nn.CrossEntropyLoss()  # loss function
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

    ## implementing default abstract functions
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        # compute privacy
        epsilon, delta, best_alpha = privacy.get_privacy_spent(
            self.epochs,
            self.num_examples["trainset"],
            self.batch_size,
            self.noise_multiplier,
            self.target_delta,
        )
        self.privacy_spent = epsilon
        print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main(
    epochs: int = 1,
    batch_size: int = 32,
    l2_norm_clip: float = 1.5,
    noise_multiplier: float = 1.5,
    learning_rate: float = 0.001,
    host: str = "[::]:8080",
) -> None:

    # Load model and data
    trainloader, testloader = load_data(batch_size)
    # Start Flower client

    client = CifarClient(
        trainloader,
        testloader,
        batch_size=batch_size,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        learning_rate=learning_rate,
    )
    fl.client.start_numpy_client(host, client=client)


if __name__ == "__main__":
    epochs = 3
    batch_size = 32
    l2_norm_clip = 1.5
    noise_multiplier = 0.5
    learning_rate = 0.001

    main(epochs, batch_size, l2_norm_clip, noise_multiplier, learning_rate)
