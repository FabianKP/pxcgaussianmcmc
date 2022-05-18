"""
Demonstrate pxcgaussianmcmc on two-dimensional test problem.
"""

from matplotlib import pyplot as plt
import numpy as np

from examples.two_dimensional_example import two_dimensional_example
from pxcgaussianmcmc import pxcgaussian, PxMALA, MYMALA


num_warmup = 10000
num_samples = 10000


def twodim_demo():
    x_map, x_true, A, b, lb = two_dimensional_example()

    # Convert ||Ax - b||^2 = (x - m).T @ P @ (x - m).
    P = A.T @ A
    m = np.linalg.solve(P, A.T @ b)


    # Get samples.
    pxmala_options = {"delta": 0.05}
    pxmala_result = pxcgaussian(SamplerType=PxMALA, num_warmup=num_warmup, num_samples=num_samples, P=P, m=m, lb=lb,
                                sampler_options=pxmala_options)
    mymala_options = {"lambda": 0.05}
    mymala_result = pxcgaussian(SamplerType=MYMALA, num_warmup=num_warmup, num_samples=num_samples, P=P, m=m, lb=lb,
                                sampler_options=mymala_options)
    pxmala_samples = pxmala_result.samples
    mymala_samples = mymala_result.samples
    print(f"PxMALA acceptance ratio: {pxmala_result.aratio}")
    print(f"MYMALA acceptance ratio: {mymala_result.aratio}")

    # Thin the samples for easier visualization.
    thinning = np.arange(0, num_samples - 1, 10)
    pxmala_samples = pxmala_samples
    mymala_samples = mymala_samples[thinning]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].axhline(0)
    ax[0].axvline(0)
    ax[0].scatter(pxmala_samples[:, 0], pxmala_samples[:, 1], s=1, label="PxMALA samples")
    ax[0].scatter(x_map[0], x_map[1], label="MAP")
    ax[0].scatter(x_true[0], x_true[1], label="Ground truth")
    ax[0].legend()

    ax[1].axhline(0)
    ax[1].axvline(0)
    ax[1].scatter(mymala_samples[:, 0], mymala_samples[:, 1], s=1, label="MYMALA samples")
    ax[1].scatter(x_map[0], x_map[1], label="MAP")
    ax[1].scatter(x_true[0], x_true[1], label="Ground truth")
    ax[1].legend()

    plt.show()


twodim_demo()