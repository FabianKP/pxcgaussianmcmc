"""
Demonstrate pxcgaussianmcmc on two-dimensional test problem.
"""

from matplotlib import pyplot as plt
import numpy as np

from examples.two_dimensional_example import two_dimensional_example
from pxcgaussianmcmc import pxcgaussian, PxMALA


num_warmup = 1000
num_samples = 10000


def twodim_demo():
    x_map, x_true, A, b, lb = two_dimensional_example()

    # Convert ||Ax - b||^2 = (x - m).T @ P @ (x - m).
    P = A.T @ A
    m = np.linalg.solve(P, A.T @ b)

    plt.figure(0)
    plt.axhline(0)
    plt.axvline(0)


    # Get samples.
    options = {"delta": 0.05}
    sampler_result = pxcgaussian(SamplerType=PxMALA, num_warmup=num_warmup, num_samples=num_samples, P=P, m=m, lb=lb,
                                 sampler_options=options)
    x_sample = sampler_result.samples
    print(f"Acceptance ratio: {sampler_result.aratio}")
    plt.scatter(x_sample[:, 0], x_sample[:, 1], s=1)

    plt.scatter(x_map[0], x_map[1], label="MAP")
    plt.scatter(x_true[0], x_true[1], label="Ground truth")
    plt.legend()

    plt.show()


twodim_demo()