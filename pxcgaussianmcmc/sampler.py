
import numpy as np

from .constrained_gaussian import ConstrainedGaussian


class Sampler:
    """
    This is the proximal MCMC sampler.
    """

    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_0: np.ndarray, options: dict):
        self._constrained_gaussian = constrained_gaussian
        # Initialize list in which samples are stored.
        self._sample_list = []
        self._acceptance_counter = 0

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
        #TODO: Implement (here).
        return 0.

    @property
    def ess(self) -> float:
        """
        Computes the effective sample size for the current list of samples. See mathematical documentation for the
        details.
        """
        #TODO: Implement (here).
        return 0.

    @property
    def acceptance_ratio(self) -> float:
        """
        Returns the acceptance ratio.
        """
        acceptance_ratio = self._acceptance_counter / len(self._sample_list)
        return acceptance_ratio
