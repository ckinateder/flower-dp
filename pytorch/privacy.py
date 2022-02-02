import logging
import math
from typing import Tuple, Generator

import flwr as fl
import numpy as np
import tensorflow as tf

import tensorflow_privacy as tfp  # for privacy calculations
import torch

logger = logging.getLogger(__name__)


def calculate_sigma_d(
    epsilon: float,
    delta: float = 1 / 2e5,
    C: float = 1.5,
    L: int = None,
    T: int = None,
    N: int = 2,
    m: int = 1e5,
) -> float:
    """Calculate sigma_d, or the std of the normal for server-side noising.
    Based off theroem 1 in 1911.00222

    Args:
        epsilon (float): measures the strength of the privacy guarantee by
            bounding how much the probability of a particular model output
            can vary by including (or excluding) a single training point.
        delta (float, optional): Bounds the probability of the privacy guarantee
            not holding. A rule of thumb is to set it to be less than the
            inverse of the size of the training dataset. Defaults to 1 / 2e5.
        C (int, optional): l2_norm_clip - clipping threshold for bounding
            weights. Defaults to 1.5.
        L (int, optional): exposures of local parameters - number of times
            local params are uploaded to server. Defaults to None, which is
            then set to value of T if given, or 3.
        T (int, optional): num rounds - number of aggregation times. Must be
            greater than or equal to L. Defaults to None, which is
            then set to value of L if given, or 3.
        N (int, optional): number of clients. Defaults to 2.
        m (int, optional): minimum size of local datasets. Defaults to 1e5.

    Returns:
        float: sigma_d

    >>> calculate_sigma_d(0.8, 1 / 1e5, 1.5, 2, 3, 2, 1e4)
    0.0009084009867385104
    """
    if L == None and T == None:
        L = 3
        T = L
    elif L == None:  # if L not given but T is
        L = T
    elif T == None:  # if T not given but L is
        T = L
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


def noise_and_clip_parameters(
    parameters: Generator, max_grad_norm: float, noise_multiplier: float
) -> None:
    """Noise and clip model parameters in place

    Args:
        parameters (Generator): torch.nn.Module.parameters()
        max_grad_norm (float): clip threshold or C value
        noise_multiplier (float): noise multiplier
    """
    std = noise_multiplier * max_grad_norm
    with torch.no_grad():
        for param in parameters:
            # clip
            param = clip_parameter(param, clip_threshold=max_grad_norm)
            # noise_multiplier: Ratio of the standard deviation (of the gaussian noise) to the clipping norm.
            param = noise_parameter(param, std=std)


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
