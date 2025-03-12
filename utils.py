import numpy as np
from typing import Callable, List


def psi_1(x: float, h: float) -> float:
    """Null to parabolic around 0 and linear outside of [-2h, 2h]"""
    if abs(x) < h:
        return 0.0
    elif abs(x) > 2 * h:
        return x
    elif x > h and x < 2 * h:
        return 8 * h - 19 * x + 14 * (x**2 / h) - 3 * x**3 / h**2
    else:
        return -8 * h - 19 * x - 14 * (x**2 / h) - 3 * x**3 / h**2


def psi(A: np.ndarray, h: float) -> np.ndarray:
    """Apply psi_1 to the entire array using np.vectorize or element-wise operation"""
    return np.vectorize(psi_1)(A, h)


def dotdot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Maths tensor operation A..B"""
    return np.sum(A * B, axis=(-2, -1))


def permut(A: np.ndarray) -> np.ndarray:
    B = np.transpose(A, (2, 3, 0, 1))
    return B


def f_of_L(L: List[float], t_per: int) -> Callable[[int], float]:
    """Returns a function from a List"""

    def f(t: int) -> float:
        return L[max(int(t / t_per), len(L) - 1)]

    return f
