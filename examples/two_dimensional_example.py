
from math import exp, sqrt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import lambertw

np.random.seed(123456)


def two_dimensional_example():
    """
    Creates two-dimensional test problem.

    :return: x_map, x_true, a, b, lb, gamma
    """
    # Prior model.
    x_bar = np.array([0., 0.])
    sigma = 0.5

    # Forward model.
    lb = np.array([0., 0.])
    x_true = np.array([1., 1.])
    fwd = np.array([[1., 4.], [8., 16.]])
    gamma = 1.
    y_bar = fwd @ x_true
    noise = gamma * np.random.randn(2)
    y = y_bar + noise

    # Compute MAP estimate by solving
    # min_x ||A x - b||_2^2 s.t. x >= lb,
    # where A = [fwd / gamma, Id / sigma ], b = [y / gamma, x_bar / sigma].
    id_n = np.identity(2)
    a = np.concatenate([fwd / (sqrt(2) * gamma), id_n / (sqrt(2) * sigma)])
    b = np.concatenate([y / (sqrt(2) * gamma), x_bar / (sqrt(2) * sigma)])
    x_map = lsq_linear(A=a, b=b, bounds=(lb, np.inf)).x

    assert np.all(x_map >= lb)

    

    return x_map, x_true, a, b
