import numpy as np
import qpsolvers
from typing import Union

from .constrained_gaussian import ConstrainedGaussian


def find_feasible_point(constrained_gaussian: ConstrainedGaussian)\
        -> np.ndarray:
    """
    Finds a point x satisfying A x = b, C x <= d, lb <= x <= ub by minimizing ||x||_2^2 subject to these constraints.

    :return: x
    :raises RuntimeError: If no feasible point could be found.
    """
    P = np.identity(constrained_gaussian.dim)
    q = np.zeros(constrained_gaussian.dim)
    # qpsolvers does not like empty matrices as arguments, so they must be converted to None using a custom function
    # "_empty_to_none".
    x_feasible = qpsolvers.solve_qp(P=P,
                                    q=q,
                                    G=_empty_to_none(constrained_gaussian.C),
                                    h=_empty_to_none(constrained_gaussian.d),
                                    A=_empty_to_none(constrained_gaussian.A),
                                    b=_empty_to_none(constrained_gaussian.b),
                                    lb=constrained_gaussian.lb,
                                    ub=constrained_gaussian.ub)
    if not constrained_gaussian.satisfies_constraints(x_feasible, tol=1e-15):
        raise RuntimeError("Was not able to find initial feasible point that satisfies all constraints.")
    return x_feasible


def _empty_to_none(mat: np.ndarray) -> Union[np.ndarray, None]:
    """
    Given a matrix, returns the original matrix if it has size > 0, or None if it has size = 0.
    """
    if mat.size == 0:
        out = None
    else:
        out = mat
    return out