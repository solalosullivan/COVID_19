import numpy as np


def psi_1(x: float, h: float):
    """Cut-off function for real numbers"""
    if abs(x) < h:
        return 0
    elif abs(x) > 2 * h:
        return x
    elif x > h and x < 2 * h:
        return 8 * h - 19 * x + 14 * (x**2 / h) - 3 * x**3 / h**2
    else:
        return -8 * h - 19 * x - 14 * (x**2 / h) - 3 * x**3 / h**2


psi = np.vectorize(psi_1, excluded=(1,))


def dotdot(A, B):
    return np.einsum("ijkl,rt->ij", A, B)


def permut(A):
    B = np.transpose(A, (2, 3, 0, 1))
    return B
