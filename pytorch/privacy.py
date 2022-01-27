import numpy as np
import torch
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def clip_parameter(
    parameter: torch.Tensor, l2_norm_clip: float, force=False
) -> torch.Tensor:
    """Clip parameter given

    >>> c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
    >>> clip_parameter(c, 5)
    tensor([[ 0.8839,  1.7678,  2.6517],
            [-0.8839,  0.8839,  3.5355]])

    Args:
        parameter (torch.Tensor): input parameter
        l2_norm_clip (float): C value
        force (bool): force norm(parameter) <= C

    Returns:
        torch.Tensor: clipped parrameter
    """
    # using formula for page 4, algorithm 1, line 7 in https://arxiv.org/pdf/1911.00222.pdf
    norm = torch.norm(parameter)

    if not norm <= l2_norm_clip and force:
        norm = l2_norm_clip
    clipped_parameter = parameter / max(1, torch.norm(parameter) / l2_norm_clip)
    return clipped_parameter


def noise_parameter(parameter: torch.Tensor, noise_multiplier: float) -> torch.Tensor:
    """Takes in clipped model parameter and noise_multiplier and adds noise

    >>> c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
    >>> np = noise_parameter(c, 0.9)
    >>> c != np
    tensor([[True, True, True],
            [True, True, True]])


    Args:
        parameter (torch.Tensor): Input parameter
        noise_multiplier (float): noise_multiplier

    Returns:
        torch.Tensor: noised parameter
    """
    noise = parameter + np.random.laplace(loc=0, scale=1 / noise_multiplier)
    return noise


def clip_and_noise(
    net: torch.nn.Module,
    l2_norm_clip: float,
    noise_multiplier: float,
) -> None:
    # clip parameters and add noise
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=l2_norm_clip)
    with torch.no_grad():
        for p in net.parameters():
            new_val = noise_parameter(new_val, noise_multiplier=noise_multiplier)
            p.copy_(new_val)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
