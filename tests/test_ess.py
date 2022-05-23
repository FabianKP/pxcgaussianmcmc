
import numpy as np
from pxcgaussianmcmc.ess import effective_sample_size


def test_ess_independent():
    m = 3       # number of chains
    n = 1000    # number of samples per chain
    d = 3       # dimension
    samples = np.random.randn(m, n, d)
    # Compute effective sample size
    ess = effective_sample_size(samples)
    print(f"ESS = {ess}, m * n = {m * n}.")
