import torch
from numpy.random import laplace


def clip_parameter(parameter: torch.Tensor, clip_threshold: float) -> torch.Tensor:
    """[summary]

    >>> c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
    >>> clip_parameter(c, 5)
    tensor([[ 0.8839,  1.7678,  2.6517],
            [-0.8839,  0.8839,  3.5355]])

    Args:
        parameter (torch.Tensor): input parameter
        clip_threshold (float): C value

    Returns:
        torch.Tensor: clipped parrameter
    """
    # using formula for page 4, algorithm 1, line 7 in https://arxiv.org/pdf/1911.00222.pdf
    clipped_parameter = parameter / max(1, torch.norm(parameter) / clip_threshold)
    return clipped_parameter


def noise_parameter(parameter: torch.Tensor, epsilon: float) -> torch.Tensor:
    noise = parameter + laplace(1 / epsilon)
    return noise


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(laplace())
