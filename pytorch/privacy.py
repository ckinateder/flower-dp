import logging
import math
from typing import Tuple, Generator, List

import numpy as np

import tensorflow_privacy as tfp  # for privacy calculations
import torch

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


def clip_parameter(parameter: torch.Tensor, clip_threshold: float) -> None:
    """Clip the parameter in place

    Args:
        parameter (torch.Tensor): input parameter
        clip_threshold (float): C value
    Returns:
        torch.Tensor: clipped parameter

    >>> c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
    >>> clip_parameter(c, 5)
    >>> c
    tensor([[ 0.8839,  1.7678,  2.6517],
            [-0.8839,  0.8839,  3.5355]])
    """
    # using formula for page 4, algorithm 1, line 7 in https://arxiv.org/pdf/1911.00222.pdf
    parameter.copy_(parameter / max(1, torch.norm(parameter) / clip_threshold))


def noise_parameter(parameter: torch.Tensor, std: float) -> None:
    """Add gaussian noise to the given parameter in place

    Args:
        parameter (torch.Tensor): input parameter
        std (float): std of the gaussian noise

    >>> std = 0.3
    >>> ntnsr = torch.Tensor([1, 4, 2, 7])
    >>> tnsr = ntnsr.clone()
    >>> noise_parameter(tnsr, std)
    """
    noise_vector = torch.normal(mean=0, std=std, size=parameter.size())
    parameter.add_(noise_vector)


def noise_and_clip_parameters(
    parameters: Generator, l2_norm_clip: float, sigma: float
) -> None:
    """Noise and clip model param GRADIENTS in place

    Args:
        parameters (Generator): torch.nn.Module.parameters()
        l2_norm_clip (float): clip threshold or C value
        sigma (float): std of normal dist

    >>> c = torch.tensor([[0.51, 0.2, 0.43], [-0.47, 0.56, -0.85]], requires_grad=True)
    >>> c = c.mean()
    >>> c.retain_grad()
    >>> c.backward()
    >>> noise_and_clip_parameters([c], 1.5, 0.05)
    """
    with torch.no_grad():
        for param in parameters:
            clip_parameter(param.grad, clip_threshold=l2_norm_clip)
            noise_parameter(param.grad, std=sigma)


def noise_weights(weights: List[np.ndarray], sigma: float) -> List[np.ndarray]:
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
