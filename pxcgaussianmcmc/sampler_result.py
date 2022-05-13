

import numpy as np


class SamplerResult:
    """
    This is a container for the result of the proximal MCMC run.

    :ivar samples: Of shape (n, d). The MCMC samples. Each row corresponds to a different sample.
    :ivar r_hat: The value of the R_hat diagnostic. See the mathematical documentation for an explanation.
    :ivar ess: The effective sample size. See the mathematical documentation for an explanation.
    """
    def __init__(self, samples: np.ndarray, r_hat: float, ess: float):
        self.samples = samples
        self.r_hat = r_hat
        self.ess = ess