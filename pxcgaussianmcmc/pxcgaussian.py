
import numpy as np
from typing import Optional

from .pxcg_sampler import PxcgSampler
from .pxc_result import PxcResult


def pxcgaussian(num_warmup: int, num_samples: int, Sigma: np.ndarray, m: np.ndarray, A: Optional[np.ndarray],
                b: Optional[np.ndarray], lb: Optional[np.ndarray], ub: Optional[np.ndarray]) -> PxcResult:
    """
    Samples from the constrained Gaussian distribution
        log p(x) = 0.5 * (x - m).T @ Sigma @ (x - m) + const.,
        truncated to the set
        A @ x = b, C @ x >= d, lb <= x <= ub.

    :param num_warmup: Number of warmup steps (aka "burnin"). These get thrown away.
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
    # Check if input makes sense.

    # Create PxcSampler object.
    sampler = PxcgSampler(Sigma=Sigma, m=m, A=A, b=b, lb=lb, ub=ub)

    # Perform warmup.
    sampler.warmup(num_warmup=num_warmup)

    # Sample.
    sampler.sample(num_samples)
    samples = sampler.samples

    # Run some convergence diagnostics.
    r_hat = sampler.r_hat
    ess = sampler.ess

    # Create PxcResult object and return it.
    result = PxcResult(samples=samples, r_hat=r_hat, ess=ess)
    return result

