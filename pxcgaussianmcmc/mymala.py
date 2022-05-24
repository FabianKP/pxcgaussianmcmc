
from math import exp, sqrt
import numpy as np
from typing import List, Optional

from .constrained_gaussian import ConstrainedGaussian
from .ldp_proximal import LDPProximal
from .sampler import Sampler


EPS = 1e-15


class MYMALA(Sampler):

    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_start: Optional[np.ndarray] = None,
                 gamma: Optional[float] = 1., delta: Optional[float] = None):
        """

        :param constrained_gaussian: An object of type pxcgaussianmcmc.ConstrainedGaussian, containing all the
            information about the target distribution.
        :param x_start: A point that satisfies all constraints. If not provided, the sampler tries to find one.
        :param gamma: The value for the regularization parameter lambda.
        :param delta: The value for the step sizes. Can either be a callable (that takes n as input) or a float.
        """
        # Call constructor of Sampler.
        Sampler.__init__(self, constrained_gaussian=constrained_gaussian, x_start=x_start)
        # Handle optional parameters.
        assert gamma > 0.
        self.gamma = gamma
        if delta is None:
            p_norm = np.linalg.norm(constrained_gaussian.P, ord=2)
            self._delta = 2 / (p_norm + 1./gamma)
            self._delta_is_const = True
        else:
            if np.isscalar(delta):
                self._delta = delta
                self._delta_is_const = True
            elif callable(delta):
                self._delta = delta
                self._delta_is_const = False
            else:
                raise ValueError("'delta' must either be scalar or callable.")
        # Initialize proximal operator.
        self._prox = LDPProximal(constrained_gaussian=constrained_gaussian)
        self._congau = constrained_gaussian
        self._P = constrained_gaussian.P
        self._m = constrained_gaussian.m
        self._acceptance_counter = 0

    def sample(self, num_samples: int):
        print(f"Sampling...")
        self._sample_list.extend(self._iterate(n=num_samples))

    def delta(self, n: int) -> float:
        if self._delta_is_const:
            return self._delta
        else:
            return self._delta(n)

    def _run_warmup(self, num_warmup: int) -> np.ndarray:
        print(f"Warmup...")
        warmup_sample_list = self._iterate(n=num_warmup)
        self._x_start = warmup_sample_list[-1]
        warmup_samples = np.array(warmup_sample_list)
        return warmup_samples

    def _iterate(self, n: int) -> List[np.ndarray]:
        """

        :param n:
        :return:
        """
        x = self._x_start
        iterates = []
        for i in range(1, n+1):
            print("\r", end="")
            print(f"Sampling: {i}/{n}.", end=" ")
            u = self._prox.evaluate(x, self.gamma)
            v = self._P @ (x - self._m)
            delta_i = self.delta(i)
            x_hat = x - delta_i * (v - (x - u) / self.gamma)
            z = np.random.randn(self.dim)
            y = x_hat + sqrt(2 * delta_i) * z
            y_satisfies_constraints = self._congau.satisfies_constraints(y, EPS)
            if not y_satisfies_constraints:
                iterates.append(x)
            else:
                u_tilde = self._prox.evaluate(y, delta_i)
                v_tilde = self._P @ (y - self._m)
                y_hat = y - delta_i * (v_tilde - (y - u_tilde) / self.gamma)
                h = 0.5 * (x - self._m).T @ self._P @ (x - self._m)
                h_tilde = 0.5 * (y - self._m).T @ self._P @ (y - self._m)
                q = (x - y_hat) @ (x - y_hat) / (4 * delta_i)
                q_tilde = (y - x_hat) @ (y - x_hat) / (4 * delta_i)
                s = - h_tilde + h - q_tilde + q
                if s > 1.:
                    # This is necessary to avoid math overflow if s is very large.
                    r = 1.
                else:
                    r = min(1., exp(s))
                eta = np.random.uniform(low=0., high=1.)
                if r >= eta:
                    iterates.append(y)
                    x = y
                    self._acceptance_counter += 1
                else:
                    iterates.append(x)
            self._sample_counter += 1

        return iterates