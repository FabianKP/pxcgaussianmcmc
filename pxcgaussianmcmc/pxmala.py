
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
        self._congau = constrained_gaussian
        self._P = constrained_gaussian.P
        self._m = constrained_gaussian.m
        self._acceptance_counter = 0

    def warmup(self, num_warmup: int):
        # Initialize
        x = self._x_start
        print("Starting warmup...")
        for i in range(num_warmup):
            print("\r", end="")
            print(f"Warmup: {i+1}/{num_warmup}.", end=" ")
            x = self._step(x)
        self._x_start = x

    def sample(self, num_sample: int):
        x = self._x_start
        print("Starting sampling...")
        for i in range(num_sample):
            print("\r", end="")
            print(f"Sampling: {i + 1}/{num_sample}.", end=" ")
            x_next = self._step(x)
            self._sample_list.append(x_next)

    def _step(self, x: np.ndarray) -> np.ndarray:
        """
        Performs one iteration of PxMALA and returns the next iterate.
        """
        xi = self._prox.evaluate(x, self._delta)
        z = np.random.randn(self._dim)
        y = xi + sqrt(2 * self._delta) * z
        # If y violates the constraints, we are already done.
        if not self._congau.satisfies_constraints(y, tol=1e-2):
            x_next = x
        else:
            zeta = self._prox.evaluate(y, self._delta)
            h = 0.5 * (x - self._m).T @ self._P @ (x - self._m)
            h_tilde = 0.5 * (y - self._m).T @ self._P @ (y - self._m)
            q = 0.25 * (x - xi) @ (x - xi) / self._delta
            q_tilde = 0.25 * (y - zeta) @ (y - zeta) / self._delta
            s = - h_tilde + h - q_tilde + q
            if s > 1.:
                # This is necessary to avoid math overflow if s is very large.
                r = 1.
            else:
                r = min(1., exp(s))
            eta = np.random.uniform(low=0., high=1.)
            if r >= eta:
                x_next = y
                self._acceptance_counter += 1
            else:
                x_next = x
        return x_next
