"""Activation functions available to CPPN nodes."""

import math


def sigmoid(x: float) -> float:
    """Return the standard logistic sigmoid.

    Returns
    -------
    float
        The logistic sigmoid of x.
    """
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x: float) -> float:
    """Hyperbolic tangent, maps values to (-1, 1).

    Returns
    -------
    float
        The hyperbolic tangent of x.
    """
    return math.tanh(x)


def sin_act(x: float) -> float:
    """Sine function, useful for generating repeating patterns.

    Returns
    -------
    float
        The sine of x.
    """
    return math.sin(x)


def gaussian(x: float) -> float:
    """Gaussian (bell curve) function.

    Returns
    -------
    float
        The Gaussian of x.
    """
    return math.exp(-(x**2))


def relu(x: float) -> float:
    """Rectified Linear Unit.

    Returns
    -------
    float
        x if x is positive, else 0.
    """
    return max(0.0, x)


ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "sin": sin_act,
    "gaussian": gaussian,
    "relu": relu,
}

DEFAULT_ACTIVATION = "sigmoid"
