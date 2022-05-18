
import numpy as np


def ess(samples: np.ndarray) -> float:
    """
    Computes effective sample size as defined in
        Vats, Flegal, Jones, "Multivariate Output Analysis for Markov Chain Monte Carlo", 2019
    
    :param samples: Of shape (n, d).
    :return: The effective sample size.
    """
    
