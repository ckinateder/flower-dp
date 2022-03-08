from collections import OrderedDict
from typing import Union

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import privacy


class PrivateClient(fl.client.NumPyClient):
    """Numpy client"""

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        model: nn.Module,
        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        epsilon: float = 10,
        delta: float = 1 / 2e5,
        l2_norm_clip: float = 1.5,
        num_rounds: int = 3,
        min_dataset_size: int = 1e5,
        batch_size: int = 32,
        epochs: int = 1,
        learning_rate: float = 0.001,
        *args,
        **kwargs,
    ) -> None:
        """Create the client
        Args:
            trainloader (DataLoader): pytorch dataloader with trainset
            testloader (DataLoader): pytorch dataloader with testset
            model (nn.Module): pytorch nn. This is an object.
            device (str, optional): device to compute on. Defaults to
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
            epsilon (float, optional): privacy budget. Defaults to 10.
            delta (float, optional): Bounds the probability of the privacy guarantee
                not holding. A rule of thumb is to set it to be less than the
                inverse of the size of the training dataset. Defaults to 1 / 2e5.
            l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
            min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
            num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
            batch_size (int, optional): Model batch size. Defaults to 32.
            epochs (int, optional): Number of train epochs. Defaults to 1.
            learning_rate (float, optional): Learning rate of optimizer. Defaults to 0.001.
        """
        super().__init__(*args, **kwargs)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.net = model.to(device)
        self.batch_size = batch_size
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
        self.learning_rate = learning_rate
        self.sigma_u = privacy.calculate_sigma_u(
            epsilon=epsilon,
            delta=delta,
            l2_norm_clip=l2_norm_clip,
            num_exposures=num_rounds,
            min_dataset_size=min_dataset_size,
        )

    def train(self) -> None:
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()  # loss function
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # put in train mode
        self.net.train()

        for _ in range(self.epochs):
            for images, labels in tqdm(self.trainloader, leave=False):
                # send to device and compute loss
                images, labels = images.to(self.device), labels.to(self.device)
                loss = criterion(self.net(images), labels)

                # compute and apply gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # apply noise
                privacy.noise_and_clip_parameters(
                    self.net.parameters(),
                    l2_norm_clip=self.l2_norm_clip,
                    sigma=self.sigma_u,
                )

    def test(self) -> Union[float, float]:
        """Validate the network on the entire test set."""
        self.net.eval()  # put in test mode
        criterion = torch.nn.CrossEntropyLoss()  # loss function
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
        # print(f"(ε = {epsilon:.2f}, δ = {delta:2e})")
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main(
    trainloader: DataLoader,
    testloader: DataLoader,
    model: nn.Module,
    epsilon: float = 10,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_rounds: int = 3,
    min_dataset_size: int = 1e5,
    batch_size: int = 32,
    epochs: int = 1,
    learning_rate: float = 0.001,
    host: str = "[::]:8080",
) -> None:
    """Create the client
    Args:
        trainloader (DataLoader): pytorch dataloader with trainset
        testloader (DataLoader): pytorch dataloader with testset
        model (nn.Module): pytorch nn. This is an object.
        epsilon (float, optional): privacy budget. Defaults to 10.
        delta (float, optional): Bounds the probability of the privacy guarantee
            not holding. A rule of thumb is to set it to be less than the
            inverse of the size of the training dataset. Defaults to 1 / 2e5.
        l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
        num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
        batch_size (int, optional): Model batch size. Defaults to 32.
        epochs (int, optional): Number of train epochs. Defaults to 1.
        learning_rate (float, optional): Learning rate of optimizer. Defaults to 0.001.
        host (str, optional): hostname and port to connect to. Defaults to "[::]:8080".
    """

    # Start Flower client
    client = PrivateClient(
        trainloader,
        testloader,
        model,
        batch_size=batch_size,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        epsilon=epsilon,
        num_rounds=num_rounds,
        delta=delta,
        min_dataset_size=min_dataset_size,
        learning_rate=learning_rate,
    )
    fl.client.start_numpy_client(host, client=client)
