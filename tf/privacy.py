from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp


def get_privacy_spent(
    epochs: int,
    num_train_examples: int,
    batch_size: int,
    noise_multiplier: float,
    target_delta: float = None,
) -> Tuple[float, float, float]:
    """Computes epsilon value for given hyperparameters.
    Based on
    github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_keras.py

    Args:
        epochs (int): Number of training epochs
        num_train_examples (int): Number of training examples
        batch_size (int): batch size
        noise_multiplier (float): noise multiplier
        target_delta (float, Optional): target delta. If given None, will
            compute to 1 / num_train_examples. Defaults to None.

    Returns:
        Tuple[float, float, float]: [description]

    >>> get_privacy_spent(1, 5e5, 32, 0.3, 2e-5)
    (12.990307141325937, 2e-05, 1.9)
    >>> get_privacy_spent(1, 5e5, 32, 0.8, 2e-06)
    (0.8856169037944343, 2e-06, 12.0)
    >>> get_privacy_spent(1, 5e5, 32, 0.8)
    (0.8856169037944343, 2e-06, 12.0)
    """

    if noise_multiplier == 0.0:
        return float("inf")
    if target_delta == None:
        target_delta = 1 / num_train_examples

    steps = epochs * num_train_examples // batch_size
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / num_train_examples
    rdp = tfp.privacy.analysis.rdp_accountant.compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )

    # Delta is set to approximate 1 / (number of training points).
    epsilon, delta, alpha = tfp.privacy.analysis.rdp_accountant.get_privacy_spent(
        orders, rdp, target_delta=target_delta
    )
    return epsilon, delta, alpha


if __name__ == "__main__":
    import doctest

    doctest.testmod()
