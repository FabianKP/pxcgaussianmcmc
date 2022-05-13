
import numpy as np
from typing import Optional


class ConstrainedGaussian:
    """
    Container object for the Gaussian distribution with density
        log p(x) = 0.5 * (L x - b).T @ Gamma @ (Lx - b) + (x - m).T @ Sigma @ (x - m) + const.,
        truncated to the set
        A @ x = b, C @ x >= d, lb <= x <= ub.
    """
    dim: int    # Dimension of x.
    Sigma: np.ndarray
    m: np.ndarray
    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray
    lb: np.ndarray
    ub: np.ndarray

    def __init__(self, Sigma: np.ndarray, m: np.ndarray, A: Optional[np.ndarray], b: Optional[np.ndarray],
                 C: Optional[np.ndarray], d: Optional[np.ndarray], lb: Optional[np.ndarray], ub: Optional[np.ndarray]):
        raise NotImplementedError
        # Check that input is consistent.

        # Set all optional parameters to their default.

        # Load them into instance variables.

        # If necessary, precompute stuff.

    def log_pdf(self, x: np.ndarray) -> float:
        """
        Returns value of log-probability density function at vector x (up to additive constant). That is, it returns
            log p(x) = 0.5 * (L x - b).T @ Gamma @ (Lx - b) + (x - m).T @ Sigma @ (x - m).
        """
        raise NotImplementedError

    def satisfies_constraints(self, x: np.ndarray, tol: float) -> bool:
        """
        Checks whether the given vector x satisfies all constraints up to a given tolerance.
        In detail, the l-infinity constraint error
        e = max(Ax - b) + max(negative_part(Cx - d)) + max(negative_part(ub - x)) + max(negative_part(x - lb))
        is computed. Returns True if e <= tol, and False if e > tol.
        """
        raise NotImplementedError
