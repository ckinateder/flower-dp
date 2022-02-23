def print_grads(params, filename: str) -> None:
    """Print the grads to file given params and filename. For debugging.
    Usage:
        print_grads(params, "original.txt")
    """
    with open(filename, "w+") as f:
        for p in params:
            f.write(str(p.grad) + "\n")


def print_params(params, filename: str) -> None:
    """Print the params to file given params and filename. For debugging"""
    with open(filename, "w+") as f:
        for p in params:
            f.write(str(p) + "\n")
