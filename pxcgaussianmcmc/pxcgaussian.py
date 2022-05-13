
import numpy as np
from typing import Optional, Type

from .constrained_gaussian import ConstrainedGaussian
from .find_feasible_point import find_feasible_point
from .sampler import Sampler
from .sampler_result import SamplerResult


def pxcgaussian(SamplerType: Type[Sampler], num_warmup: int, num_samples: int, P: np.ndarray, m: np.ndarray,
                A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None,
                d: Optional[np.ndarray] = None, lb: Optional[np.ndarray] = None, ub: Optional[np.ndarray] = None,
                sampler_options: dict = None) -> SamplerResult:
    """
    Samples from the constrained Gaussian distribution
        log p(x) = 0.5 * (x - m).T @ P @ (x - m) + const.,
        truncated to the set
        A @ x = b, C @ x <= d, lb <= x <= ub.

    :param SamplerType: A derived class of the abstract Sampler class. Possible types are:
        - PxMCMC: Proximal MCMC.
        - MYULA: Moreau-Yosida unadjusted Langevin algorithm.
    :param num_warmup: Number of warmup steps (aka "burnin"). These get thrown away.
    :param num_samples: Desired number of samples.
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
    :param sampler_options: Dictionary with sampler-specific options. See documentation
        of the individual samplers for all options.
    :returns result: An object of type pxcgaussianmcmc.PxcResult.
    """
    # Check if input makes sense.
    if sampler_options is None:
        sampler_options = {}

    # Creates ConstrainedGaussian object.
    constrained_gaussian = ConstrainedGaussian(P=P, m=m, A=A, b=b, C=C, d=d, lb=lb, ub=ub)

    # Find a feasible point.
    x_0 = find_feasible_point(constrained_gaussian)

    # Create PxcSampler object.
    sampler = SamplerType(constrained_gaussian, x_0=x_0, options=sampler_options)

    # Perform warmup.
    sampler.warmup(num_warmup=num_warmup)

    # Sample.
    sampler.sample(num_samples)
    samples = sampler.samples

    # Run some convergence diagnostics.
    r_hat = sampler.r_hat
    ess = sampler.ess
    ratio = sampler.acceptance_ratio

    # Create PxcResult object and return it.
    result = SamplerResult(samples=samples, r_hat=r_hat, ess=ess, aratio=ratio)
    return result
