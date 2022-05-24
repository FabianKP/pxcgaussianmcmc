
import numpy as np
import qpsolvers
from typing import Optional

from .empty_to_none import empty_to_none


EPS = 1e-15


class ConstrainedGaussian:
    """
    Container object for the Gaussian distribution with density
        log p(x) = (x - m).T @ P @ (x - m) + const.,
        truncated to the set
        A @ x = b, C @ x <= d, lb <= x <= ub.
    """
    dim: int    # Dimension of x.
    P: np.ndarray
    m: np.ndarray
    A: np.ndarray
    b: np.ndarray
    C: np.ndarray
    d: np.ndarray
    lb: np.ndarray
    ub: np.ndarray

    def __init__(self, P: np.ndarray, m: np.ndarray, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
                 C: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None, lb: Optional[np.ndarray] = None,
                 ub: Optional[np.ndarray] = None):
        """

        :param P: Of shape (d, d). The PRECISION matrix (i.e. inverse of covariance).
        Must be a symmetric positive definite matrix.
        :param m: Of shape (d, ).
        :param A: Of shape (m, d).
        :param b: Of shape (m, ).
        :param C: Of shape (l, d).
        :param d: Of shape (l, ).
        :param lb: Of shape (d, ). Setting an entry to - np.inf means that the corresponding coordinate is unbounded from
            below.
        :param ub: Of shape (d, ). Setting an entry to np.inf means that the corresponding coordinate is unbounded from
            above.
        """
        # Check that input is consistent.
        assert P.ndim == 2, "Sigma must be a matrix."
        dim = P.shape[0]
        assert P.shape == (dim, dim), "Sigma must be square."
        assert m.shape == (dim, ), "Shape of m does not match Sigma."
        for mat, vec, str1, str2 in zip([A, C], [b, d], ["A", "C"], ["b", "d"]):
            assert (mat is None and vec is None) or (mat is not None and vec is not None), \
                f"Must provide both {str1} and {str2} or neither."
            if mat is not None:
                assert mat.shape[1] == dim, f"Shape of {str1} must match Sigma."
                assert vec.shape == (mat.shape[0], ), f"Shapes of {str1} and {str2} must match."
        for vec, name in zip([lb, ub], ["lb", "ub"]):
            if vec is not None:
                assert vec.shape == (dim, ), f"{name} must have shape ({dim}, )."

        # Set all unspecified parameters to their default values.
        if lb is None:
            lb = - np.infty * np.ones(dim)
        if ub is None:
            ub = np.infty * np.ones(dim)
        if A is None:
            A = np.ndarray((0, dim))
        if b is None:
            b = np.ndarray((0, 0))
        if C is None:
            C = np.ndarray((0, dim))
        if d is None:
            d = np.ndarray((0, 0))

        # Load them into instance variables.
        self.dim = dim
        self.P = P
        self.m = m
        self.A = A
        self.b = b
        self.C = C
        self.d = d
        self.lb = lb
        self.ub = ub

    def log_pdf(self, x: np.ndarray) -> float:
        """
        Returns value of log-probability density function at vector x (up to additive constant). That is, it returns
            log p(x) = (x - m).T @ Sigma @ (x - m).
        """
        return 0.5 * (x - self.m).T @ self.P @ (x - self.m)

    def satisfies_constraints(self, x: np.ndarray, tol: float) -> bool:
        """
        Checks whether the given vector x satisfies all constraints up to a given tolerance.
        In detail, the l-infinity constraint error
        e = max(Ax - b) + max(positive(d - Cx)) + max(positive(x - ub)) + max(positive(lb - x))
        is computed. Returns True if e <= tol, and False if e > tol.
        """
        equality_error = np.max(self.A @ x - self.b, initial=0.)
        # Evaluate np.max with initial=0. so that in the case of an empty array it returns 0.
        inequality_error = np.max((self.d - self.C @ x).clip(min=0.), initial=0.)
        upper_bound_error = np.max((x - self.ub).clip(min=0.))
        lower_bound_error = np.max((self.lb - x).clip(min=0.))
        constraint_error = equality_error + inequality_error + upper_bound_error + lower_bound_error

        satisfied = (constraint_error <= tol)
        return satisfied
