import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
import privacy
from typing import Union
from tqdm import tqdm


class PrivateClient(fl.client.NumPyClient):
    """Numpy client"""

    def __init__(
        self,
        trainset: tf.data.Dataset,
        testset: tf.data.Dataset,
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
        self.loss_function = keras.losses.SparseCategoricalCrossentropy()
        self.net = tf.keras.applications.MobileNetV2(
            trainset.element_spec[0].shape[1:],
            classes=10,
            weights=None,
        )
        self.net.compile(self.optimizer, self.loss_function, metrics=["accuracy"])
        ###

    def clip_and_noise_gradients(self, gradients: list) -> list:
        clipped_grads = privacy.clip_gradients(gradients, self.l2_norm_clip)
        noised_grads = privacy.noise_gradients(clipped_grads, self.sigma_u)
        return noised_grads

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.net(x, training=True)
            loss_value = self.loss_function(y, logits)
        grads = tape.gradient(loss_value, self.net.trainable_weights)
        grads_prime = self.clip_and_noise_gradients(grads)
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
    trainset: tf.data.Dataset,
    testset: tf.data.Dataset,
    epsilon: float = 10,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_rounds: int = 3,
    min_dataset_size: int = 1e5,
    epochs: int = 1,
    host: str = "[::]:8080",
) -> None:
    """Create the client

    Args:
        trainset (tf.data.Dataset): tf dataset for training
        testset (tf.data.Dataset): tf dataset for testing
        epsilon (float, optional): privacy budget. Defaults to 10.
        delta (float, optional): Bounds the probability of the privacy guarantee
            not holding. A rule of thumb is to set it to be less than the
            inverse of the size of the training dataset. Defaults to 1 / 2e5.
        l2_norm_clip (float, optional): Euclidian norm clip for gradients. Defaults to 1.5.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.
        num_rounds (int, optional): num rounds - number of aggregation times. Defaults to 3.
        epochs (int, optional): Number of train epochs. Defaults to 1.
    """

    # Start Flower client
    client = PrivateClient(
        trainset,
        testset,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        epsilon=epsilon,
        num_rounds=num_rounds,
        delta=delta,
        min_dataset_size=min_dataset_size,
    )
    fl.client.start_numpy_client(host, client=client)


if __name__ == "__main__":
    # privacy guarantees for (epsilon, delta)-DP
    epsilon = 0.8  # lower is better
    delta = 1 / 2e5
    l2_norm_clip = 1.5  # max euclidian norm of the weight gradients

    # client variables
    epochs = 1  # how many epochs to go through
    batch_size = 256  # batch size for training
    learning_rate = 0.001  # how quickly the model learns
    min_dataset_size = 1e5  # minimum training set size

    # server variables
    num_rounds = 3  # number of train/val rounds to go through
    min_available_clients = 3  # minimum number of clients to train/val - `N``
    clients_per_round = 3  # number of clients to be selected for each round - `K`
    # K <= N
    host = "[::]:8080"

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    trainset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=1024)
        .batch(256)
    )
    testset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .shuffle(buffer_size=1024)
        .batch(256)
    )

    main(
        trainset,
        testset,
        epsilon,
        delta,
        l2_norm_clip,
        num_rounds,
        min_dataset_size,
        epochs,
        host,
    )
