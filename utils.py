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


def psi(A: np.ndarray, h: float):
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = psi_1(A[i, j], h)
    return C


# def dotdot(A, B):
#     return np.einsum("ijkl,rt->ij", A, B)


def dotdot(A, B):
    C = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(B.shape[0]):
                for l in range(B.shape[1]):
                    C[i, j] += A[i, j, k, l] * B[k, l]
    return C


def permut(A):
    B = np.transpose(A, (2, 3, 0, 1))
    return B


def f_of_L(L, t_per):
    def f(t):
        return L[max(int(t / t_per), len(L) - 1)]

    return f
