
from math import exp, sqrt
import numpy as np
from typing import List

from .constrained_gaussian import ConstrainedGaussian
from .proximal_operator import ProximalOperator
from .sampler import Sampler


EPS = 1e-15


class MYMALA(Sampler):
    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_0: np.ndarray, options: dict):
        """

        :param constrained_gaussian:
        :param x_0:
        :param options: Additional sampler options.
            - lambda: The value for the regularization parameter lambda.
            - delta: The value for the stepsizes. Can either be a callable (that takes n as input) or a float.
        """
        assert constrained_gaussian.satisfies_constraints(x_0, tol=EPS)
        # Read in options.
        delta = options.setdefault("delta", 1.)
        lam = options.setdefault("lambda", delta)
        assert lam > 0.
        self.lam = lam
        self._dim = constrained_gaussian.dim
        if isinstance(delta, float):
            self._delta_const = delta
            self._delta_fun = None
        else:
            self._delta_const = None
            self._delta_fun = delta

        # Call constructor of Sampler.
        Sampler.__init__(self, constrained_gaussian=constrained_gaussian, x_0=x_0, options=options)
        self._x_start = x_0
        # Initialize proximal operator.
        raise NotImplementedError("WARNING: For MYMALA, I have to use a different proximal operator (no Sigma-term),"
                                  "since I perform splitting.")
        self._prox = ProximalOperator(constrained_gaussian=constrained_gaussian)
        self._congau = constrained_gaussian
        self._P = constrained_gaussian.P
        self._m = constrained_gaussian.m
        self._acceptance_counter = 0

    def warmup(self, num_warmup: int):
        warmup_samples = self._iterate(n=num_warmup)
        self._x_start = warmup_samples[-1]

    def sample(self, num_samples: int):
        self._samples = self._iterate(n=num_samples)

    def delta(self, n: int) -> float:
        if self._delta_const is None:
            return self._delta_fun(n)
        else:
            return self._delta_const

    def _iterate(self, n: int) -> List[np.ndarray]:
        """

        :param n:
        :return:
        """
        x = self._x_start
        iterates = []
        for i in range(1, n+1):
            u = self._prox.evaluate(x, self.lam)
            v = self._P @ (x - self._m)
            delta_i = self.delta(i)
            x_hat = x - delta_i * (v - (x - u) / self.lam)
            z = np.random.randn(self._dim)
            y = x_hat + sqrt(2 * delta_i) * z
            y_satisfies_constraints = self._congau.satisfies_constraints(y, EPS)
            if not y_satisfies_constraints:
                iterates.append(x)
            else:
                u_tilde = self._prox.evaluate(y, delta_i)
                v_tilde = self._P @ (y - self._m)
                y_hat = y - delta_i * (v_tilde - (y - u_tilde) / self.lam)
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
        return iterates


