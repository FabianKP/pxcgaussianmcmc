import numpy as np

from .constrained_gaussian import ConstrainedGaussian


def find_feasible_point(constrained_gaussian: ConstrainedGaussian)\
        -> np.narray:
    """
    Finds a point x satisfying A x = b, C x >= d, lb <= x <= ub.

    :return: x
    """
    raise NotImplementedError