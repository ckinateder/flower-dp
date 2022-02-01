import os

import flwr as fl
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

import privacy

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int,
        epochs: int,
        l2_norm_clip: float,
        noise_multiplier: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # assign params
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.privacy_spent = None
        self.num_examples = x_train.shape[0]
        self.target_delta = 1 / self.num_examples
        # init model
        self.build_model()

    def build_model(self):
        # sequential model
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    3,
                    padding="same",
                    input_shape=self.x_train.shape[1:],
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        # differentially private optimizer
        optimizer = tfp.VectorizedDPKerasSGDOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=1,
            learning_rate=0.001,
        )
        # loss function
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model.compile(optimizer, loss=loss, metrics=["accuracy"])

    ## implementing default abstract functions
    def get_parameters(self):  # type: ignore
        return self.model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size
        )
        # compute privacy
        epsilon, delta, best_alpha = privacy.get_privacy_spent(
            self.epochs,
            self.num_examples,
            self.batch_size,
            self.noise_multiplier,
            self.target_delta,
        )
        self.privacy_spent = epsilon
        print(f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(x_test), {"accuracy": accuracy}


if __name__ == "__main__":
    # Load and compile Keras model
    epochs = 1
    batch_size = 32
    l2_norm_clip = 1.5
    noise_multiplier = 0.3

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Start Flower client

    client = CifarClient(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=batch_size,
        epochs=epochs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
