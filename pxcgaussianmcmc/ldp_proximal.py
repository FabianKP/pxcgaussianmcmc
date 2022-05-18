
import numpy as np
import qpsolvers

from .constrained_gaussian import ConstrainedGaussian
from .empty_to_none import empty_to_none
from .proximal_operator import ProximalOperator


class LDPProximal(ProximalOperator):
    """
    Given a constrained Gaussian, solves the problem
    min_xi ||xi - x||^2
    s.t. A xi = b, C xi >= d, l <= xi <= u.
    """
    def __init__(self, constrained_gaussian: ConstrainedGaussian):
        self._dim = constrained_gaussian.dim
        self._con_gau = constrained_gaussian

    def evaluate(self, x: np.ndarray, delta: float) -> np.ndarray:
        """
        Returns the minimizer xi of the proximal optimization problem
            min_z 0.5 * ||z - x||^2
            s.t. A z = b, C z >= d, l <= z <= u.
        :param x:
        :param delta:
        :return: xi
        """
        # 0.5 * (z - x)^2 = 0.5 * z @ z - x @ z + z @ z,
        # -> P = Id, q = - x.
        # Solve problem using qpsolvers.
        P = np.identity(self._dim)
        q = -x
        z = qpsolvers.solve_qp(P=P,
                               q=q, G=empty_to_none(self._con_gau.C),
                               h=empty_to_none(self._con_gau.d),
                               A=empty_to_none(self._con_gau.A),
                               b=empty_to_none(self._con_gau.b),
                               lb=self._con_gau.lb,
                               ub=self._con_gau.ub)
        # Check that xi satisfies constraints.
        if not self._con_gau.satisfies_constraints(z, tol=1e-5):
            print("WARNING: Constraint violated.")
        return z
