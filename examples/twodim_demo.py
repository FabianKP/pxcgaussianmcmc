"""
Demonstrate pxcgaussianmcmc on two-dimensional test problem.
"""

from matplotlib import pyplot as plt
import numpy as np

from examples.two_dimensional_example import two_dimensional_example
from pxcgaussianmcmc import ConstrainedGaussian, PxMALA, MYMALA, sufficient_sample_size


num_warmup = 10000
num_samples = 10000


def twodim_demo():
    x_map, x_true, A, b, lb = two_dimensional_example()

    # Convert ||Ax - b||^2 = (x - m).T @ P @ (x - m).
    P = A.T @ A
    m = np.linalg.solve(P, A.T @ b)

    # Create ConstrainedGaussian.
    test_distribution = ConstrainedGaussian(P=P, m=m, lb=lb)

    # Initialize sampler.
    pxmala_sampler = PxMALA(constrained_gaussian=test_distribution, delta=0.1)
    # Warmup.
    pxmala_sampler.warmup(num_warmup)
    # Sample.
    pxmala_sampler.sample(num_samples)
    # Get result.
    pxmala_result = pxmala_sampler.get_result()
    # Get samples.
    pxmala_samples = pxmala_result.samples
    # Assess convergence.
    ess_min = sufficient_sample_size(dim=2, alpha=0.05, epsilon=0.1)
    print(f"Sufficient effective sample size: {ess_min}.")
    print(f"Effective sample size of PxMALA: {pxmala_result.ess}. R_hat = {pxmala_result.r_hat}")
    print(f"Acceptance ratio for PxMALA: {pxmala_result.aratio}.")

    # Now, repeat with MYMALA.
    mymala_sampler = MYMALA(constrained_gaussian=test_distribution, gamma=0.05)
    mymala_sampler.warmup(num_warmup)
    mymala_sampler.sample(num_samples)
    mymala_result = mymala_sampler.get_result()
    mymala_samples = mymala_result.samples
    print(f"Effective sample size of MYMALA: {mymala_result.ess}. R_hat = {pxmala_result.r_hat}")
    print(f"Acceptance ratio for MYMALA: {mymala_result.aratio}.")

    # MAKE PLOTS:

    # Thin the samples for easier visualization.
    thinning = np.arange(0, num_samples - 1, 10)
    pxmala_samples = pxmala_samples
    mymala_samples = mymala_samples[thinning]
    # And plot:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_xlim(-0.1, 1.8)
    ax[0].set_ylim(-0.1, 1.8)
    ax[0].axhline(0)
    ax[0].axvline(0)
    ax[0].scatter(pxmala_samples[:, 0], pxmala_samples[:, 1], s=1, label="PxMALA samples")
    ax[0].scatter(x_map[0], x_map[1], label="MAP")
    ax[0].scatter(x_true[0], x_true[1], label="Ground truth")
    ax[0].legend()
    ax[1].set_xlim(-0.1, 1.8)
    ax[1].set_ylim(-0.1, 1.8)
    ax[1].axhline(0)
    ax[1].axvline(0)
    ax[1].scatter(mymala_samples[:, 0], mymala_samples[:, 1], s=1, label="MYMALA samples")
    ax[1].scatter(x_map[0], x_map[1], label="MAP")
    ax[1].scatter(x_true[0], x_true[1], label="Ground truth")
    ax[1].legend()
    plt.show()


twodim_demo()