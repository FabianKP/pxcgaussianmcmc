
import numpy as np


def ess(samples: np.ndarray) -> float:
    """
    Computes effective sample size as defined in
        Vats, Flegal, Jones, "Multivariate Output Analysis for Markov Chain Monte Carlo", 2019
    
    :param samples: Of shape (n, d).
    :return: The effective sample size.
    """
    # Compute sample covariance of all chains combined.
    # Combine multivariate replicated lugsail batch means estimator:
    #   For batch size b and b / 3:
    #       Separate chain into batches.
    #       Compute batch means.
    #       Compute T_b.
    #   Compute T_L
    # Evaluate ESS.

