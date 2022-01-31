import numpy as np
import torch
import tensorflow as tf
import logging
import math
import flwr as fl

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


def noise_weights(
    weights: fl.common.typing.Weights, sigma_d: float
) -> fl.common.typing.Weights:
    weights = weights.copy()
    for i in range(len(weights)):
        weights[i] += np.random.normal(scale=sigma_d)
    return weights


if __name__ == "__main__":
    import doctest

    doctest.testmod()
