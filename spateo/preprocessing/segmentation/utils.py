import cv2
import numpy as np
from kneed import KneeLocator


def circle(k: int) -> np.ndarray:
    """Draw a circle of diameter k.

    Args:
        k: Diameter

    Returns:
        8-bit unsigned integer Numpy array with 1s and 0s
    """
    r = (k - 1) // 2
    return cv2.circle(np.zeros((k, k), dtype=np.uint8), (r, r), r, 1, -1)


def knee(X: np.ndarray) -> float:
    """Find the knee point of an arbitrary array.

    Args:
        X: Numpy array of values

    Returns:
        Knee
    """
    unique, counts = np.unique(X, return_counts=True)
    argsort = np.argsort(unique)
    x = unique[argsort]
    y = counts[argsort] / X.size

    kl = KneeLocator(x, y, curve="concave")
    return kl.knee
