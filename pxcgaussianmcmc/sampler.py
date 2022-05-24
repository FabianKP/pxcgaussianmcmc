
import numpy as np
from typing import Optional

from .constrained_gaussian import ConstrainedGaussian
from .convergence_diagnostics import effective_sample_size, r_hat
from .find_feasible_point import find_feasible_point
from .sampler_result import SamplerResult


EPS = 1e-15


class Sampler:
    """
    This is the proximal MCMC sampler.
    """

    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_start: Optional[np.ndarray] = None):
        """

        :param constrained_gaussian: An object of type pxcgaussianmcmc.ConstrainedGaussian, containing all the
            information about the target distribution.
        :param x_start: A point that satisfies all constraints. If not provided, the sampler tries to find one.
        """
        self._constrained_gaussian = constrained_gaussian
        self.dim = constrained_gaussian.dim
        # Initialize list in which samples are stored.
        self._sample_list = []
        self._acceptance_counter = 0
        self._sample_counter = 0

        # Check that x_start satisfies all constraints, or try to find a feasible point.
        if x_start is None:
            self._x_start = find_feasible_point(constrained_gaussian)
        else:
            if constrained_gaussian.satisfies_constraints(x_start, tol=EPS):
                self._x_start = x_start
            else:
                raise ValueError("'x_start' does not satisfy all constraints.")

    def get_result(self) -> SamplerResult:
        """
        The current array of samples. This is either an empty array, if there are no samples yet, or of shape (n, d),
        where n is the number of samples.
        """
        assert len(self._sample_list) == self._sample_counter
        sample_arr = np.array(self._sample_list)
        return self._samples_to_result(sample_arr)

    def warmup(self, num_warmup: int) -> SamplerResult:
        """
        Performs burn-in.

        :param num_warmup: Number of burnin-steps.
        :returns result: Returns pxcgaussianmcmc.SamplerResult object.
        """
        warmup_samples = self._run_warmup(num_warmup)
        warmup_result = self._samples_to_result(warmup_samples)
        # Reset counters.
        self._acceptance_counter = 0
        self._sample_counter = 0
        return warmup_result

    def sample(self, num_sample: int):
        """
        Runs the chain a desired number of steps and appends the results to self._sample_list

        :param num_sample: Desired number of steps.
        """
        raise NotImplementedError

    def _run_warmup(self, num_warmup: int) -> np.ndarray:
        raise NotImplementedError

    def _samples_to_result(self, samples: np.ndarray) -> SamplerResult:
        r = r_hat(samples)
        ess = effective_sample_size(samples)
        aratio = self._compute_acceptance_ratio()
        result = SamplerResult(samples=samples, r_hat=r, ess=ess, aratio=aratio)
        return result

    def _compute_acceptance_ratio(self) -> float:
        """
        Returns the acceptance ratio.
        """
        acceptance_ratio = self._acceptance_counter / self._sample_counter
        return acceptance_ratio
