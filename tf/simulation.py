import os
from multiprocessing import Process
from pickletools import optimize
from typing import List, Union
from tensorflow import keras
import copy
import tensorflow as tf

import client
import server


def load_cifar10_data(batch_size: int) -> Union[tf.data.Dataset, tf.data.Dataset]:
    """Load cifar10 data
    Args:
        batch_size (int): batch size to be applied to the dataloaders
    Returns:
        Union[tf.data.Dataset, tf.data.Dataset]: training dataloader, testing dataloader
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    trainset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=1024)
        .batch(batch_size)
    )
    testset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .shuffle(buffer_size=1024)
        .batch(batch_size)
    )
    return trainset, testset


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

    # load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

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
                    (x_train, y_train),
                    (x_test, y_test),
                    epsilon,
                    delta,
                    l2_norm_clip,
                    num_rounds,
                    min_dataset_size,
                    epochs,
                    host,
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
