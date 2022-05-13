
import numpy as np

from .constrained_gaussian import ConstrainedGaussian


class ProximalOperator:
    """
    Given a constrained Gaussian, solves the problem
    min_xi (xi - m).T @ Sigma @ (xi - m) + (1/delta) * ||xi - x||^2
    s.t. A xi = b, C xi >= d, l <= xi <= u.
    """
    def __init__(self, constrained_gaussian: ConstrainedGaussian):
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, delta: float) -> np.ndarray:
        """
        Returns the minimizer xi of the proximal optimization problem
            min_xi (xi - m).T @ Sigma @ (xi - m) + (1/delta) * ||xi - x||^2
            s.t. A xi = b, C xi >= d, l <= xi <= u.
        :param x:
        :param delta:
        :return: xi
        """
        raise NotImplementedError
