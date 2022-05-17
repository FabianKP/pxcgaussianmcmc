
from math import exp, sqrt
import numpy as np

from .constrained_gaussian import ConstrainedGaussian
from .proximal_operator import ProximalOperator
from .sampler import Sampler
from .sampler import Sampler


class MYMALA(Sampler):
    def __init__(self, constrained_gaussian: ConstrainedGaussian, x_0: np.ndarray, options: dict):
        
        # Call constructor of Sampler.
        Sampler.__init__(self, constrained_gaussian=constrained_gaussian, x_0=x_0, options=options)