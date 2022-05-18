
import numpy as np


class ProximalOperator:
    """
    Abstract base class for Proximal Operator.
    """

    def evaluate(self, x: np.ndarray, delta: float) -> np.ndarray:
        raise NotImplementedError


