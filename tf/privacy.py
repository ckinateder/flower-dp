import numpy as np
import tensorflow as tf
from typing import Tuple
import tensorflow_privacy as tfp


def get_privacy_spent(
    epochs: int,
    num_train_examples: int,
    batch_size: int,
    noise_multiplier: float,
    target_delta: float,
) -> Tuple[float, float]:
    """Computes epsilon value for given hyperparameters.

    Based on
    github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_keras.py
    """
    if noise_multiplier == 0.0:
        return float("inf")
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
    return epsilon, alpha
