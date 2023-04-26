import math, random
from typing import List


def make_zero_array(m: int, n: int) -> List[List[float]]:
    """Creates an m x n array filled with zero values

    Args:
        m - first dimension
        n - second dimension

    Returns:
        m x n array of zeros
    """
    return [[0.0] * n for _ in range(m)]


def make_random_array(
    m: int, n: int, lower: float = -2.0, upper: float = 2.0
) -> List[List[float]]:
    """Creates an m x n array filled with random values

    Args:
        m - first dimension
        n - second dimension

    Returns:
        m x n array of random values
    """
    return [[random.uniform(lower, upper) for _ in range(n)] for _ in range(m)]


# Two activation functions, sigmoid & tanh, and their derivatives. Activation functions
# take the raw output of a node and essentially determine if it fires.  They often tend
# to compress the range of output in some way for cleaner gradients and input to
# subsequent layers.


def sigmoid(x: float) -> float:
    """Computes 1/(1+e^-x)

    Args:
        x - the value to apply the sigmoid too (in our neural net corresponds to the
            output of a single node)

    Returns:
        the result of the function applied to x
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except:
        return 0.0


def d_sigmoid(y: float) -> float:
    """Computes the derivative of sigmoid, based on the value of the function

    Args:
        y - output of the sigmoid function

    Returns:
        derivative of sigmoid
    """
    return y * (1.0 - y)


def tanh(x: float) -> float:
    """Computes the hyperbolic tangent of x

    Args:
        x - the value to apply the hyperbolic tangent too (in our neural net corresponds
            to the output of a single node)

    Returns:
        the result of the function applied to x
    """
    return math.tanh(x)


def d_tanh(y: float) -> float:
    """Computes the derivative of tanh, based on the value of the function

    Args:
        y - output of the tanh function

    Returns:
        derivative of tanh
    """
    return 1.0 - y * y


class SizeMismatch(Exception):
    """A class to represent an error when the wrong number of input values is offered

    Attributes:
        desired - expected number of input values
        actual - actual number of input values received
    """

    def __init__(self, desired: int, actual: int) -> None:
        """Simple constructor setting given attributes

        Args:
            desired - expected number of input values
            actual - actual number of input values received
        """
        self.desired = desired
        self.actual = actual

    def __str__(self) -> str:
        """String representation of error"""
        return f"Incorrect number of inputs: {self.desired} required, {self.actual} received"
