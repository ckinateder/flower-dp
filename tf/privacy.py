import logging
import math
from typing import List, Tuple, Optional

import numpy as np
import flwr as fl

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


def _noise_weights(weights: List[np.ndarray], sigma: float) -> List[np.ndarray]:
    """Noise flower weights. Weights will be noised with individual drawings
    from the normal - i.e. if weights are an array with shape (1, 5), there
    will be 5 unique drawings from the normal.

    Args:
        weights (List[np.ndarray]): list of numpy arrays - weights
        sigma (float): std of normal distribution

    Returns:
        List[np.ndarray]: noised copy of weights
    """
    weights = weights.copy()
    for i in range(len(weights)):
        weights[i] += np.random.normal(scale=sigma, size=weights[i].shape)
    return weights


def noise_aggregated_weights(
    aggregated_weights: Optional[fl.common.Parameters], sigma: float
) -> Optional[fl.common.Parameters]:
    """Extension of noise_weights to be used with aggregate params on the server

    Args:
        aggregated_weights (Optional[fl.common.Parameters]): weights
        sigma (float): std of normal distribution

    Returns:
        Optional[fl.common.Parameters]: noised weights or None if None given
    """

    # add noise
    if aggregated_weights is not None:
        noised_weights = list(aggregated_weights)  # make into list so assignable
        for i in range(len(aggregated_weights)):
            if type(aggregated_weights[i]) == fl.common.typing.Parameters:
                weights = fl.common.parameters_to_weights(aggregated_weights[i])
                weights = _noise_weights(weights, sigma)
                noised_weights[i] = weights  # reassign parameters
    return tuple(noised_weights)
