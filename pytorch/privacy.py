import logging
import math
from typing import Tuple

import flwr as fl
import numpy as np
import tensorflow as tf

import tensorflow_privacy as tfp
import torch

logger = logging.getLogger(__name__)


def calculate_sigma_d(
    epsilon: float,
    delta: float = 1 / 2e5,
    L: float = 3,
    C: float = 1.5,
    T: float = 3,
    N: float = 2,
    m: float = 1e5,
) -> float:
    """Calculate sigma_d, or the std of the normal for server-side noising.
    Based off theroem 1 in 1911.00222

    Args:
        epsilon (float): privacy budget
        delta (float, optional): delta value. Defaults to 1 / 2e5.
        L (float, optional): exposures of local parameters. Defaults to 3.
        C (float, optional): l2_norm_clip - clipping threshold for bounding weights. Defaults to 1.5.
        T (float, optional): num rounds - number of aggregation times. Defaults to 3.
        N (float, optional): number of clients. Defaults to 2.
        m (float, optional): minimum size of local datasets. Defaults to 1e5.

    Returns:
        float: sigma_d

    >>> calculate_sigma_d(0.8, 1 / 1e5, 2, 1.5, 3, 2, 1e4)
    0.0009084009867385104
    """
    if T <= L * (N ** 0.5):
        return 0
    # for epsilon (0, 1) - need to verify for outside bound
    c = (2 * math.log(1.25 / delta)) ** 0.5
    return (2 * c * C * (((T ** 2) - ((L ** 2) * N)) ** 0.5)) / (m * N * epsilon)


def clip_parameter(parameter: torch.Tensor, clip_threshold: float) -> torch.Tensor:
    """Clip the parameter

    Args:
        parameter (torch.Tensor): input parameter
        clip_threshold (float): C value
    Returns:
        torch.Tensor: clipped parrameter

    >>> c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
    >>> clip_parameter(c, 5)
    tensor([[ 0.8839,  1.7678,  2.6517],
            [-0.8839,  0.8839,  3.5355]])
    """
    # using formula for page 4, algorithm 1, line 7 in https://arxiv.org/pdf/1911.00222.pdf
    clipped_parameter = parameter / max(1, torch.norm(parameter) / clip_threshold)
    return clipped_parameter


def noise_parameter(parameter: torch.Tensor, std: float) -> torch.Tensor:
    """Add gaussian noise to the given parameter

    Args:
        parameter (torch.Tensor): input parameter
        std (float): std of the gaussian noise

    Returns:
        torch.Tensor: noised parameter

    >>> std = 0.3
    >>> tnsr = torch.Tensor([1, 4, 2, 4])
    >>> ntnsr = noise_parameter(tnsr, std)
    >>> noise = tnsr - ntnsr
    >>> all([x <= std for x in noise])
    True
    """
    noised_parameter = parameter + np.random.normal(scale=std)
    return noised_parameter


def noise_weights(
    weights: fl.common.typing.Weights, sigma_d: float
) -> fl.common.typing.Weights:
    weights = weights.copy()
    for i in range(len(weights)):
        weights[i] += np.random.normal(scale=sigma_d)
    return weights


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
