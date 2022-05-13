import numpy as np
import qpsolvers
from typing import Union

from .constrained_gaussian import ConstrainedGaussian
from .empty_to_none import empty_to_none


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
                                    G=empty_to_none(constrained_gaussian.C),
                                    h=empty_to_none(constrained_gaussian.d),
                                    A=empty_to_none(constrained_gaussian.A),
                                    b=empty_to_none(constrained_gaussian.b),
                                    lb=constrained_gaussian.lb,
                                    ub=constrained_gaussian.ub)
    if not constrained_gaussian.satisfies_constraints(x_feasible, tol=1e-15):
        raise RuntimeError("Was not able to find initial feasible point that satisfies all constraints.")
    return x_feasible