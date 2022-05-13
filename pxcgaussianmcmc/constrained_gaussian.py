
import numpy as np
from typing import Optional


class ConstrainedGaussian:
    """
    Container object for the Gaussian distribution with density
        log p(x) = (x - m).T @ Sigma @ (x - m) + const.,
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
        # Check that input is consistent.
        assert Sigma.ndim != 2, "Sigma must be a matrix."
        dim = Sigma.shape[0]
        assert Sigma.shape != (dim, dim), "Sigma must be square."
        assert m.shape != (dim, ), "Shape of m does not match Sigma."
        for mat, vec, str1, str2 in zip([A, C], [b, d], ["A", "C"], ["b", "d"]):
            assert (mat is None and vec is None) or (mat is not None and vec is not None), \
                f"Must provide both {str1} and {str2} or neither."
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
            A = np.array((0, dim))
        if b is None:
            b = np.array([])
        if C is None:
            C = np.array((0, dim))
        if d is None:
            d = np.array([])

        # Load them into instance variables.
        self.Sigma = Sigma
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
        return 0.5 * (x - self.m).T @ self.Sigma @ (x - self.m)

    def satisfies_constraints(self, x: np.ndarray, tol: float) -> bool:
        """
        Checks whether the given vector x satisfies all constraints up to a given tolerance.
        In detail, the l-infinity constraint error
        e = max(Ax - b) + max(positive(d - Cx)) + max(positive(x - ub)) + max(positive(lb - x))
        is computed. Returns True if e <= tol, and False if e > tol.
        """
        equality_error = np.max(self.A @ x - self.b)
        inequality_error = np.max((self.d - self.C @ x).clip(min=0.))
        upper_bound_error = np.max((x - self.ub).clip(min=0.))
        lower_bound_error = np.max((self.lb - x).clip(min=0.))
        constraint_error = equality_error + inequality_error + upper_bound_error + lower_bound_error

        satisfied = (constraint_error <= tol)
        return satisfied
