
from pxcgaussianmcmc.constrained_gaussian import ConstrainedGaussian
from pxcgaussianmcmc.find_feasible_point import find_feasible_point

import numpy as np


def test_find_feasible_point():
    np.random.seed(12345)
    dim = 8
    a = np.random.randn(2, 8)
    b = np.random.randn(2)
    lb = np.zeros(8)
    Sigma = np.identity(dim)
    m = np.zeros(dim)
    congau = ConstrainedGaussian(P=Sigma, m=m, A=a, b=b, lb=lb)

    x_feasible = find_feasible_point(constrained_gaussian=congau)
    assert congau.satisfies_constraints(x=x_feasible, tol=1e-15)

