
import numpy as np
import qpsolvers

from .constrained_gaussian import ConstrainedGaussian
from .empty_to_none import empty_to_none


class ProximalOperator:
    """
    Given a constrained Gaussian, solves the problem
    min_xi (xi - m).T @ Sigma @ (xi - m) + (1/delta) * ||xi - x||^2
    s.t. A xi = b, C xi >= d, l <= xi <= u.
    """
    def __init__(self, constrained_gaussian: ConstrainedGaussian):
        self._dim = constrained_gaussian.dim
        self._con_gau = constrained_gaussian

    def evaluate(self, x: np.ndarray, delta: float) -> np.ndarray:
        """
        Returns the minimizer xi of the proximal optimization problem
            min_z (xi - m).T @ P @ (xi - m) + (1/delta) * ||xi - x||^2
            s.t. A xi = b, C xi >= d, l <= xi <= u.
        :param x:
        :param delta:
        :return: xi
        """
        # Since qp_solver solves problems of the form 0.5 x.T @ P @ x + q @ x, we need to out-multiply:
        #   (xi - m).T @ P @ (xi - m) + (1/delta) * (xi - x).T (xi - x) =
        # = 0.5 xi.T @ 2 * (P + I / delta) @ xi - 2 * (P.T @ m + x / delta).T @ xi + const.
        # Solve problem using qpsolvers.
        P = 2 * (self._con_gau.P + np.identity(self._dim) / delta) # Careful: Clash of notations (P and P).
        q = - 2 * (self._con_gau.P.T @ self._con_gau.m + x / delta)
        xi = qpsolvers.solve_qp(P=P,
                                q=q,
                                G=empty_to_none(self._con_gau.C),
                                h=empty_to_none(self._con_gau.d),
                                A=empty_to_none(self._con_gau.A),
                                b=empty_to_none(self._con_gau.b),
                                lb=self._con_gau.lb,
                                ub=self._con_gau.ub)
        # Check that xi satisfies constraints.
        if not self._con_gau.satisfies_constraints(xi, tol=1e-5):
            print("WARNING: Constraint violated.")
        return xi


