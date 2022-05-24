
import numpy as np
from pxcgaussianmcmc.convergence_diagnostics import effective_sample_size, r_hat, sufficient_sample_size


def test_ess_independent():
    np.random.seed(12345)
    m = 3       # number of chains
    n = 1000    # number of samples per chain
    d = 3       # dimension
    rtol = 0.2
    samples = np.random.randn(m, n, d)
    # Compute effective sample size
    ess = effective_sample_size(samples)
    assert (1 - rtol) * m * n <= ess <= (1 + rtol) * m * n


def test_r_hat():
    np.random.seed(12345)
    m = 3  # number of chains
    n = 1000  # number of samples per chain
    d = 3  # dimension
    rtol = 0.01
    samples = np.random.randn(m, n, d)
    # Compute r_hat.
    r = r_hat(samples)
    assert 1 <= r <= 1 + rtol

def test_w_value():
    alpha = 0.05
    eps = 0.1
    d = 3
    w = sufficient_sample_size(d, alpha, eps)

    print(f"W({d},{alpha},{eps}) = {w}.")


def test_w_large():
    alpha = 0.01
    eps = 0.1
    d = 636
    w = sufficient_sample_size(d, alpha, eps)

    print(f"W({d},{alpha},{eps}) = {w}.")

