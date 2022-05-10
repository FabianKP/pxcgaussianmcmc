
import numpy as np
from typing import Optional


class PxcResult:
    """
    This is a container for the result of the proximal MCMC run.

    :ivar samples: Of shape (n, d). The MCMC samples. Each row corresponds to a different sample.
    :ivar r_hat: The value of the R_hat diagnostic. See the mathematical documentation for an explanation.
    :ivar ess: The effective sample size. See the mathematical documentation for an explanation.
    """
    def __int__(self, samples: np.ndarray, r_hat: float, ess: float):
        self.samples = samples
        self.r_hat = r_hat
        self.ess = ess


def pxcgaussian(warmup: int, num_samples: int, Sigma: np.ndarray, m: np.ndarray, A: Optional[np.ndarray],
                b: Optional[np.ndarray], lb: Optional[np.ndarray], ub: Optional[np.ndarray]) -> PxcResult:
    """
    Samples from the constrained Gaussian distribution
        log p(x) = 0.5 * (x - m).T @ Sigma @ (x - m) + const.,
        truncated to the set
        A @ x = b, C @ x >= d, lb <= x <= ub.

    :param warmup: Number of warmup steps (aka "burnin"). These get thrown away.
    :param num_samples: Desired number of samples.
    :param Sigma: Of shape (d, d). Must be a symmetric positive definite matrix.
    :param m: Of shape (d, ).
    :param A: Of shape (m, d).
    :param b: Of shape (m, ).
    :param lb: Of shape (d, ). Setting an entry to - np.inf means that the corresponding coordinate is unbounded from
        below.
    :param ub: Of shape (d, ). Setting an entry to np.inf means that the corresponding coordinate is unbounded from
        above.
    :returns result: An object of type pxcgaussianmcmc.PxcResult.
    """

    # Create PxcSampler object.

    # Sample.

    # Run some convergence diagnostics.

    # Create PxcResult object and return it.

