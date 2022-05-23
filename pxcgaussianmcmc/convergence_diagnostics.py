
from math import sqrt, pi, pow
import numpy as np
from typing import Optional
from scipy.stats import chi2
from scipy.special import gamma


def effective_sample_size(samples: np.ndarray, b: Optional[int] = None) -> float:
    """
    Computes effective sample size as defined in
        Vats, Flegal, Jones, "Multivariate Output Analysis for Markov Chain Monte Carlo", 2019

    :param samples: Of shape (n, d) or (m, n, d), corresponding to m chains of n d-dimensional samples.
    :param b: The batch size. If not provided, this is chosen equal to n^(1/2) (if n <= 1000) or n^(1/3) (if n > 1000).
    :return: The effective sample size.
    """
    assert samples.ndim in [2, 3]
    # Reshape if samples is of shape (n, d).
    if samples.ndim == 2:
        samples = samples.reshape((1, samples.shape[0], samples.shape[1]))
    # Compute sample covariance of all chains combined.
    combined_chain = samples.reshape(-1, samples.shape[2])
    Sigma = np.cov(combined_chain.T)
    # Compute replicated lugsail batch means estimator.
    T_L = replicated_lugsail(samples, b)

    # Make some sanity checks.
    d = combined_chain.shape[1]
    assert Sigma.shape == (d, d)
    assert T_L.shape == (d, d)

    # Evaluate ESS.
    detSigma = np.linalg.norm(Sigma)
    detT = np.linalg.norm(T_L)
    if np.isclose(detT, 0):
        raise RuntimeError("T_L is singular.")
    mn = combined_chain.shape[0]
    detSigma_detT = detSigma / detT
    ess = mn * np.power(detSigma_detT, 1 / d)

    return ess


def r_hat(samples: np.ndarray, b: Optional[int] = None) -> float:
    """
    Computes multivariate R_hat statistic as defined in
        Vats, Flegal, Jones, "Multivariate Output Analysis for Markov Chain Monte Carlo", 2019:
    R_hat = sqrt((n-1) / n + m / ess)

    :param samples: Of shape (n, d) or (m, n, d), corresponding to m chains of n d-dimensional samples.
    """
    assert samples.ndim in [2, 3]
    # Reshape if samples is of shape (n, d).
    if samples.ndim == 2:
        samples = samples.reshape((1, samples.shape[0], samples.shape[1]))
    m = samples.shape[0]
    n = samples.shape[1]

    R_hat_squared = (n - 1) / n + m / effective_sample_size(samples, b)
    R_hat = sqrt(R_hat_squared)

    return R_hat


def w_value(d: int, alpha: float, epsilon: float):
    """
    Computes W(alpha, epsilon, d) (see mathematical documentation).

    :param alpha:
    :param epsilon:
    :param d:
    :return:
    """
    # Check input.
    assert 0 <= alpha <= 1.
    assert 0 <= epsilon
    assert d >= 1
    # Compute W(d, alpha, epsilon)
    chi2_alpha_d = chi2.ppf(q=1 - alpha, df=d)
    gamma_d_half = gamma(d / 2)
    numerator = pow(2, (2 / d)) * pi * chi2_alpha_d
    denominator = pow(d * gamma_d_half, 2 / d) * epsilon * epsilon
    w = numerator / denominator

    assert w > 0
    return w


def replicated_lugsail(samples: np.ndarray, b: Optional[int] = None) -> np.ndarray:
    """
    Computes the multivariate replicated lugsail batch means estimator.

    :param samples: Of shape (m, n, d).
    :return: Of shape (d, d).
    """
    n = samples.shape[1]
    # Pick default batch size, if not provided by user.
    if b is None:
        if n <= 1000:
            b = np.floor(np.sqrt(n)).astype(int)
        else:
            b = np.floor(n ** (1/3)).astype(int)
    T_b_list = []
    for batch_size in [b, np.ceil(b / 3)]:
        batch_means_list = []
        for chain in samples:
            # For batch size b and b / 3:
            # Separate chain into batches.
            num_batches = np.ceil(n / batch_size).astype(int)
            batches = np.array_split(chain, num_batches)
            # Compute batch means.
            batch_means_m = [np.mean(batch, axis=0) for batch in batches]
            batch_means_list.extend(batch_means_m)
        batch_means = np.array(batch_means_list)
        # Compute T_b.
        T_b = batch_size * np.cov(batch_means.T)
        T_b_list.append(T_b)
    # Compute T_L
    T_L = 2 * T_b_list[0] - T_b_list[1]

    return T_L