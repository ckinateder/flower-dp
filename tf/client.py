import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import privacy
from typing import Union


class PrivateClient(fl.client.NumPyClient):
    """Numpy client"""

    def __init__(
        self,
        trainset: tf.data.Dataset,
        testset: tf.data.Dataset,
        model: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        loss_function: keras.losses.Loss,
        epsilon: float = 10,
        delta: float = 1 / 2e5,
        l2_norm_clip: float = 1.5,
        num_rounds: int = 3,
        min_dataset_size: int = 1e5,
        epochs: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """Create the client

        Args:
            trainset (tf.data.Dataset): tf dataset for training
            testset (tf.data.Dataset): tf dataset for testing
            model (keras.Model): model
            optimizer (keras.optimizers.Optimizer): optmizer
            loss_function (keras.losses.Loss): loss function
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
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.trainset = trainset
        self.testset = testset
        self.net = model
        self.epochs = epochs
        self.l2_norm_clip = l2_norm_clip
        self.epsilon = epsilon
        """
        self.num_examples = {
            "trainset": len(trainset.dataset),
            "testset": len(testset.dataset),
            "total": len(trainset.dataset) + len(testset.dataset),
        }  # stored in a dictionary
        """
        self.num_examples = {
            "trainset": len(x_train),
            "testset": len(x_test),
            "total": len(x_train) + len(x_test),
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
        self.optimizer = optimizer

    def train(self) -> None:
        """Train self.net on the training set."""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.net.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)

    def test(self) -> Union[float, float]:
        """Validate the network on the entire test set."""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        loss, accuracy = self.net.evaluate(x_test, y_test)
        return loss, accuracy

    def get_parameters(self):
        return self.net.get_weights()

    def set_parameters(self, parameters):
        self.net.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


fl.client.start_numpy_client(
    "[::]:8080", client=PrivateClient(None, None, None, None, None)
)
