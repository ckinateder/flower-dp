import os, sys

sys.path.insert(0, os.getcwd())

from collections import OrderedDict
from typing import Generator, Union

import flwr as fl
import privacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class PrivateClient(fl.client.NumPyClient):
    """Numpy client"""

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        model: nn.Module,
        loss_function: nn.Module,
        epsilon: float = 10,
        delta: float = 1 / 2e5,
        l2_norm_clip: float = 1.5,
        num_rounds: int = 3,
        min_dataset_size: int = 1e5,
        epochs: int = 1,
        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        *args,
        **kwargs,
    ) -> None:
        """Create the client
        Args:
            trainloader (DataLoader): pytorch dataloader with trainset
            testloader (DataLoader): pytorch dataloader with testset
            model (nn.Module): pytorch nn. This is an object.
            loss_function (nn.Module): Loss function.
            device (str, optional): device to compute on. Defaults to
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
            epsilon (float, optional): privacy budget. Defaults to 10.
            delta (float, optional): Bounds the probability of the privacy guarantee
                not holding. A rule of thumb is to set it to be less than the
                inverse of the size of the training dataset. Defaults to 1 / 2e5.
            l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
            min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
            num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
            epochs (int, optional): Number of train epochs. Defaults to 1.
        """
        super().__init__(*args, **kwargs)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.net = model.to(device)
        self.epochs = epochs
        self.l2_norm_clip = l2_norm_clip
        self.epsilon = epsilon
        self.num_examples = {
            "trainset": len(trainloader.dataset),
            "testset": len(testloader.dataset),
            "total": len(trainloader.dataset) + len(testloader.dataset),
        }  # stored in a dictionary
        self.privacy_spent = None
        self.delta = delta
        self.sigma_u = privacy.calculate_sigma_u(
            epsilon=epsilon,
            delta=delta,
            l2_norm_clip=l2_norm_clip,
            num_exposures=num_rounds,
            min_dataset_size=min_dataset_size,
        )
        self.loss_function = loss_function
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def train_step(self, x, y) -> None:
        # send to device and compute loss
        images, labels = x.to(self.device), y.to(self.device)
        loss = self.loss_function(self.net(images), labels)

        # compute grads
        self.optimizer.zero_grad()
        loss.backward()

        # apply noise
        with torch.no_grad():
            for param in self.net.parameters():
                param.grad.copy_(
                    torch.Tensor(
                        privacy.noise_and_clip_param(
                            param.grad,
                            l2_norm_clip=self.l2_norm_clip,
                            sigma=self.sigma_u,
                        )
                    )
                )

        # apply gradients
        self.optimizer.step()

    def train(self) -> None:
        """Train self.net on the training set."""
        self.net.train()  # put in train mode

        for _ in range(self.epochs):
            for images, labels in tqdm(self.trainloader, leave=False):
                self.train_step(images, labels)

    def test(self) -> Union[float, float]:
        """Validate the network on the entire test set."""
        self.net.eval()  # put in test mode
        criterion = self.loss_function
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
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
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main(
    trainloader: DataLoader,
    testloader: DataLoader,
    model: nn.Module,
    loss_function: nn.Module,
    epsilon: float = 10,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_rounds: int = 3,
    min_dataset_size: int = 1e5,
    epochs: int = 1,
    host: str = "[::]:8080",
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> None:
    """Create the client
    Args:
        trainloader (DataLoader): pytorch dataloader with trainset
        testloader (DataLoader): pytorch dataloader with testset
        model (nn.Module): pytorch nn. This is an object.
        loss_function (nn.Module): Loss function.
        device (str, optional): device to compute on. Defaults to
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        epsilon (float, optional): privacy budget. Defaults to 10.
        delta (float, optional): Bounds the probability of the privacy guarantee
            not holding. A rule of thumb is to set it to be less than the
            inverse of the size of the training dataset. Defaults to 1 / 2e5.
        l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
        num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
        epochs (int, optional): Number of train epochs. Defaults to 1.
        host (str, optional): hostname and port to connect to. Defaults to "[::]:8080".
    """

    # Start Flower client
    client = PrivateClient(
        trainloader,
        testloader,
        model,
        loss_function,
        device=device,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        epsilon=epsilon,
        num_rounds=num_rounds,
        delta=delta,
        min_dataset_size=min_dataset_size,
    )
    fl.client.start_numpy_client(host, client=client)
