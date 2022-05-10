
import numpy as np
from typing import Optional, Union


class PxcgSampler:
    """
    This is the proximal MCMC sampler.
    """

    def __init__(self, Sigma: np.ndarray, m: np.ndarray, A: Optional[np.ndarray],
                b: Optional[np.ndarray], lb: Optional[np.ndarray], ub: Optional[np.ndarray]):
        self._Sigma = Sigma
        self._m = m
        self._A = A
        self._b = b
        self._lb = lb
        self._ub = ub
        # Initialize list in which samples are stored.
        self._sample_list = []

    @property
    def samples(self) -> np.ndarray:
        """
        The current array of samples. This is either an empty array, if there are no samples yet, or of shape (n, d),
        where n is the number of samples.
        """
        return np.array(self._sample_list)

    def warmup(self, num_warmup: int):
        """
        Performs burn-in.

        :param num_warmup: Number of burnin-steps.
        """
        raise NotImplementedError

    def sample(self, num_sample):
        """
        Runs the chain a desired number of steps and appends the results to self._sample_list

        :param num_sample: Desired number of steps.
        """
        raise NotImplementedError

    @property
    def r_hat(self) -> float:
        """
        Computes the R_hat statistic for the current list of samples. See mathematical documentation for the details.
        """
        raise NotImplementedError

    @property
    def ess(self) -> float:
        """
        Computes the effective sample size for the current list of samples. See mathematical documentation for the
        details.
        """
        raise NotImplementedError
