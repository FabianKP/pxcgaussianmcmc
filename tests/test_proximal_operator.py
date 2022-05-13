
import numpy as np
from pxcgaussianmcmc.proximal_operator import ProximalOperator
from pxcgaussianmcmc.constrained_gaussian import ConstrainedGaussian


def test_init():
    dim =  20
    Sigma = np.identity(dim)
    m = np.zeros(dim)
    lb = np.zeros(dim)
    congau = ConstrainedGaussian(Sigma=Sigma, m=m, lb=lb)
    proxy = ProximalOperator(constrained_gaussian=congau)


def test_unconstrained():
    """
    Let's check that for the unconstrained problem, the cost-gradient is close to zero.
    """
    dim = 2
    np.random.seed(12345)
    L = np.random.randn(dim, dim)
    Sigma = L @ L.T
    m = np.random.randn(dim)
    delta = 0.123
    congau = ConstrainedGaussian(Sigma=Sigma, m=m)
    proxy = ProximalOperator(constrained_gaussian=congau)

    x = np.random.randn(dim)
    z = proxy.evaluate(x=x, delta=delta)
    # The cost gradient at z is simply 2 * Sigma (z - m) + 2 * (z - x) / delta.
    grad_z = 2 * Sigma @ (z - m) + 2 * (z - x) / delta
    norm_grad_z = np.linalg.norm(grad_z)

    assert norm_grad_z <= 1e-6