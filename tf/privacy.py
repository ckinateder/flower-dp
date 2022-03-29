import logging
import math
from typing import Generator, List, Optional, Tuple
import tensorflow as tf
import flwr as fl
import numpy as np
from flwr.common.typing import Parameters, Scalar, Weights

logger = logging.getLogger(__name__)


def calculate_c(delta: float) -> float:
    """Calculate c value based on section IIC in 1911.00222

    Args:
        delta (float): delta value

    Returns:
        float: c value

    >>> calculate_c(2 / 1e5)
    4.699557816587533
    """
    return (2 * math.log(1.25 / delta)) ** 0.5


def calculate_sigma_d(
    epsilon: float,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_exposures: int = None,
    num_rounds: int = None,
    num_clients: int = 2,
    min_dataset_size: int = 1e5,
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
        l2_norm_clip (int, optional): l2_norm_clip - clipping threshold for gradients. Defaults to 1.5.
        num_exposures (int, optional): exposures of local parameters - number of times
            local params are uploaded to server. Defaults to None, which is
            then set to value of num_rounds if given, or 3.
        num_rounds (int, optional): num rounds - number of aggregation times. Must be
            greater than or equal to num_exposures. Defaults to None, which is
            then set to value of num_exposures if given, or 3.
        num_clients (int, optional): number of clients. Defaults to 2.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.

    Returns:
        float: sigma_d

    >>> calculate_sigma_d(0.8, 1 / 1e5, 1.5, 2, 3, 2, 1e4)
    0.0009084009867385104
    """
    T = num_rounds
    L = num_exposures
    N = num_clients
    m = min_dataset_size
    C = l2_norm_clip
    if L == None and T == None:
        L = 3
        T = L
    elif L == None:  # if L not given but T is
        L = T
    elif T == None:  # if T not given but L is
        T = L
    if T <= L * (N ** 0.5):
        return 0
    c = calculate_c(delta)
    return (2 * c * C * (((T ** 2) - ((L ** 2) * N)) ** 0.5)) / (m * N * epsilon)


def calculate_sigma_u(
    epsilon: float,
    delta: float = 1 / 2e5,
    l2_norm_clip: float = 1.5,
    num_exposures: int = 3,
    min_dataset_size: int = 1e5,
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
        l2_norm_clip (int, optional): l2_norm_clip - clipping threshold for gradients. Defaults to 1.5.
        num_exposures (int, optional): exposures of local parameters - number of times
            local params are uploaded to server. Defaults to 3.
        min_dataset_size (int, optional): minimum size of local datasets. Defaults to 1e5.

    Returns:
        float: sigma_d

    >>> calculate_sigma_d(0.8, 1 / 1e5, 1.5, 2, 3, 2, 1e4)
    0.0009084009867385104
    """
    L = num_exposures
    m = min_dataset_size
    C = l2_norm_clip
    c = calculate_c(delta)

    return (2 * C * c * L) / (epsilon * m)


def clip_gradients(gradients: list, clip_threshold: float) -> list:
    """Clip the gradients

    Args:
        parameter (list): list of gradients
        clip_threshold (float): C value
    Returns:
        list: clipped list of grads
    """
    # using formula for page 4, algorithm 1, line 7 in https://arxiv.org/pdf/1911.00222.pdf
    return [tf.clip_by_norm(g, clip_threshold) for g in gradients]


def noise_gradients(gradients: list, std: float) -> list:
    """noise the gradients

    Args:
        parameter (list): list of gradients
        std (float): std of the gaussian noise

    Returns:
        list: clipped list of grads
    """
    return [tf.random.normal(g.shape, stddev=std) for g in gradients]


def noise_weights(weights: Weights, sigma: float) -> Weights:
    """Noise flower weights. Weights will be noised with individual drawings
    from the normal - i.e. if weights are an array with shape (1, 5), there
    will be 5 unique drawings from the normal.
    Args:
        weights (Weights): list of numpy arrays - weights
        sigma (float): std of normal distribution of noise to be added
    Returns:
        Weights: noised copy of weights
    """
    weights = weights.copy()
    for i in range(len(weights)):
        weights[i] += np.random.normal(scale=sigma, size=weights[i].shape)
    return weights


def server_side_noise(parameters: Parameters, sigma: float) -> Optional[Parameters]:
    """Apply noise_weights to flower parameters

    Args:
        parameters (Parameters): server params
        sigma (float): std of normal distribution of noise to be added

    Returns:
        Optional[Parameters]: noised weights
    """
    noised_parameters = None
    if parameters is not None:
        weights = fl.common.parameters_to_weights(parameters)
        weights = noise_weights(weights, sigma=sigma)  # noise weights
        noised_parameters = fl.common.weights_to_parameters(weights)
    return noised_parameters


if __name__ == "__main__":
    import doctest

    doctest.testmod()
