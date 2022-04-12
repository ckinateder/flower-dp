import os, sys

sys.path.insert(0, os.getcwd())
from typing import Union

import flwr as fl
import numpy as np
import privacy
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


class PrivateClient(fl.client.NumPyClient):
    """Numpy client"""

    def __init__(
        self,
        trainset: Union[np.ndarray, np.ndarray],
        testset: Union[np.ndarray, np.ndarray],
        net: tf.keras.Model,
        loss_function: tf.keras.losses.Loss,
        epsilon: float = 10,
        delta: float = 1 / 2e5,
        l2_norm_clip: float = 1.5,
        num_rounds: int = 3,
        min_dataset_size: int = 1e5,
        epochs: int = 1,
        batch_size: int = 256,
        *args,
        **kwargs,
    ) -> None:
        """Create the client

        Args:
            trainset (Union[np.ndarray, np.ndarray]): (x_train, y_train) numpy arrays
            testset (Union[np.ndarray, np.ndarray]): (x_test, y_test) numpy arrays
            net (tf.keras.Model): model
            loss_function (tf.keras.losses.Loss): loss function
            epsilon (float, optional): privacy budget. Defaults to 10.
            delta (float, optional): Bounds the probability of the privacy guarantee
                not holding. A rule of thumb is to set it to be less than the
                inverse of the size of the training dataset. Defaults to 1 / 2e5.
            l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
            min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
            num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
            epochs (int, optional): Number of train epochs. Defaults to 1.
            batch_size (int, optional): Batch size to use. Defaults to 256.
        """
        super().__init__(*args, **kwargs)

        trainset = (
            tf.data.Dataset.from_tensor_slices(trainset)
            .shuffle(buffer_size=1024)
            .batch(batch_size)
        )
        testset = (
            tf.data.Dataset.from_tensor_slices(testset)
            .shuffle(buffer_size=1024)
            .batch(batch_size)
        )
        self.trainset = trainset
        self.testset = testset
        self.epochs = epochs
        self.l2_norm_clip = l2_norm_clip
        self.epsilon = epsilon
        self.num_examples = {
            "trainset": len(self.trainset),
            "testset": len(self.testset),
            "total": len(self.trainset) + len(self.testset),
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

        ###
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_function = loss_function
        self.net = net
        self.net.compile(self.optimizer, self.loss_function, metrics=["accuracy"])
        ###

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.net(x, training=True)
            loss_value = self.loss_function(y, logits)
        grads = tape.gradient(loss_value, self.net.trainable_weights)

        # apply noise
        grads_prime = [
            tf.clip_by_norm(g, self.l2_norm_clip)
            for g in [g + tf.random.normal(g.shape, stddev=self.sigma_u) for g in grads]
        ]

        self.optimizer.apply_gradients(zip(grads_prime, self.net.trainable_weights))
        return loss_value

    def train(self) -> None:
        """Train self.net on the training set."""
        for epoch in range(self.epochs):
            for _, (x_batch_train, y_batch_train) in tqdm(
                enumerate(self.trainset),
                total=self.num_examples["trainset"],
                leave=False,
            ):
                self.train_step(x_batch_train, y_batch_train)

    def test(self) -> Union[float, float]:
        """Validate the network on the entire test set."""
        loss, accuracy = self.net.evaluate(self.testset)
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


def main(
    trainset: Union[np.ndarray, np.ndarray],
    testset: Union[np.ndarray, np.ndarray],
    net: tf.keras.Model,
    loss_function: tf.keras.losses.Loss,
    epsilon: float = 10,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_rounds: int = 3,
    min_dataset_size: int = 1e5,
    epochs: int = 1,
    host: str = "[::]:8080",
    batch_size: int = 256,
) -> None:
    """Create the client

    Args:
        trainset (Union[np.ndarray, np.ndarray]): (x_train, y_train) numpy arrays
        testset (Union[np.ndarray, np.ndarray]): (x_test, y_test) numpy arrays
        net (tf.keras.Model): model
        loss_function (tf.keras.losses.Loss): loss function
        epsilon (float, optional): privacy budget. Defaults to 10.
        delta (float, optional): Bounds the probability of the privacy guarantee
            not holding. A rule of thumb is to set it to be less than the
            inverse of the size of the training dataset. Defaults to 1 / 2e5.
        l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
        num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
        epochs (int, optional): Number of train epochs. Defaults to 1.
        batch_size (int, optional): Batch size to use. Defaults to 256.
    """

    # Start Flower client
    client = PrivateClient(
        trainset,
        testset,
        net,
        loss_function,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        epsilon=epsilon,
        num_rounds=num_rounds,
        delta=delta,
        min_dataset_size=min_dataset_size,
        batch_size=batch_size,
    )
    fl.client.start_numpy_client(host, client=client)
