from math import factorial

import numpy as np
from numpy.linalg import matrix_power


def my_c2d(A, B, dt):
    """
    Discretizes matrix A, B by numerically computing exp(A * dt) and B * exp(A * dt)
    :param A: state matrix as ndarray
    :param B: input matrix as ndarray
    :param dt: sampling time
    :return: discretized representations of A and B as ndarray
    """
    Ad = np.eye(A.shape[0]) + A * dt
    Bd = B * dt
    prev_A = A
    k = 2
    eps = 1e-8
    above_treshold = True
    while above_treshold:
        f_term = 1 / factorial(k)
        Bd_term = f_term * np.matmul(prev_A, B) * dt ** k
        Bd += Bd_term
        prev_A = matrix_power(A, k)
        Ad_term = f_term * prev_A * dt ** k
        Ad += Ad_term
        k += 1
        if np.sum(Ad_term) < eps and np.sum(Bd_term) < eps:
            above_treshold = False

    return Ad, Bd
