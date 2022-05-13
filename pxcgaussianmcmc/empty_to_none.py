
import numpy as np
from typing import Union


def empty_to_none(mat: np.ndarray) -> Union[np.ndarray, None]:
    """
    Given a matrix, returns the original matrix if it has size > 0, or None if it has size = 0.
    """
    if mat.size == 0:
        out = None
    else:
        out = mat
    return out