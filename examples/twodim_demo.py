"""
Demonstrate pxcgaussianmcmc on two-dimensional test problem.
"""

from matplotlib import pyplot as plt
import numpy as np

from examples.two_dimensional_example import two_dimensional_example


def twodim_demo():
    x_map, x_true, a, b = two_dimensional_example()

    plt.figure(0)
    plt.axhline(0)
    plt.axvline(0)

    plt.scatter(x_map[0], x_map[1], label="MAP")
    plt.scatter(x_true[0], x_true[1], label="Ground truth")

    # create some pseudosamples
    x_sample = np.random.randn(100, 2)
    plt.scatter(x_sample[:, 0], x_sample[:, 1], s=3)
    plt.legend()

    plt.show()


twodim_demo()