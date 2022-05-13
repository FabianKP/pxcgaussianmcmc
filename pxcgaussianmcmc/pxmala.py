
from math import exp, sqrt
import numpy as np

from .constrained_gaussian import ConstrainedGaussian
from .proximal_operator import ProximalOperator
from .sampler import Sampler


EPS = 1e-15


class PxMALA(Sampler):

    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_0: np.ndarray, options: dict):
        """
        :param constrained_gaussian: Container for the constrained Gaussian distribution.
        :param x_0: A feasible point.
        :param options: Dictionary with further sampler options.
        """
        # Check input for consistency.
        assert constrained_gaussian.satisfies_constraints(x_0, tol=EPS)
        # Load delta.
        delta = options.setdefault("delta", 1.)
        if delta <= 0.:
            raise ValueError("'delta' must be strictly positive.")
        self._delta = delta
        self._dim = constrained_gaussian.dim
        # Call constructor of Sampler.
        Sampler.__init__(self, constrained_gaussian=constrained_gaussian, x_0=x_0, options=options)
        self._x_start = x_0
        # Initialize proximal operator.
        self._prox = ProximalOperator(constrained_gaussian=constrained_gaussian)
        self._Sigma = constrained_gaussian.Sigma
        self._m = constrained_gaussian.m

    def warmup(self, num_warmup: int):
        # Initialize
        x = self._x_start
        for i in range(num_warmup):
            x = self._step(x)
        self._x_start = x

    def sample(self, num_sample: int):
        x = self._x_start
        for i in range(num_sample):
            x_next = self._step(x)
            self._sample_list.append(x_next)

    def _step(self, x: np.ndarray) -> np.ndarray:
        """
        Performs one iteration of PxMALA and returns the next iterate.
        """
        xi = self._prox.evaluate(x, self._delta)
        z = np.random.randn(self._dim)
        y = xi + sqrt(2 * self._delta) * z
        zeta = self._prox.evaluate(y, self._delta)
        h = 0.5 * (x - self._m).T @ self._Sigma @ (x - self._m)
        h_bar = 0.5 * (y - self._m).T @ self._Sigma @ (y - self._m)
        q = 0.25 * (x - xi) @ (x - xi) / self._delta
        q_bar = 0.25 * (y - zeta) @ (y - zeta) / self._delta
        s = h - h_bar + q - q_bar
        r = min(1., exp(s))
        eta = np.random.uniform(low=0., high=1.)
        if r >= eta:
            x_next = y
        else:
            x_next = x
        return x_next
