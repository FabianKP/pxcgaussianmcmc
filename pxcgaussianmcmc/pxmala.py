
from math import exp, sqrt
import numpy as np
from typing import Optional

from .constrained_gaussian import ConstrainedGaussian
from .cls_proximal import CLSProximal
from .sampler import Sampler


EPS = 1e-15


class PxMALA(Sampler):

    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_start: Optional[np.ndarray] = None,
                 delta: Optional[float] = 1.):
        """
        :param constrained_gaussian: An object of type pxcgaussianmcmc.ConstrainedGaussian, containing all the
            information about the target distribution.
        :param x_start: A point that satisfies all constraints. If not provided, the sampler tries to find one.
        :param delta: The stepsize.
        """
        Sampler.__init__(self, constrained_gaussian=constrained_gaussian, x_start=x_start)
        if delta <= 0.:
            raise ValueError("'delta' must be strictly positive.")
        self._delta = delta
        # Initialize proximal operator.
        self._prox = CLSProximal(constrained_gaussian=constrained_gaussian)
        self._congau = constrained_gaussian
        self._P = constrained_gaussian.P
        self._m = constrained_gaussian.m
        self._acceptance_counter = 0

    def _run_warmup(self, num_warmup: int) -> np.ndarray:
        # Initialize
        x = self._x_start
        print("Starting warmup...")
        sample_list = []
        for i in range(1, num_warmup + 1):
            print("\r", end="")
            print(f"Warmup: {i}/{num_warmup}.", end=" ")
            x = self._step(x)
            sample_list.append(x)
        self._x_start = x
        samples = np.array(sample_list)
        return samples

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
        z = np.random.randn(self.dim)
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
        self._sample_counter += 1
        return x_next
