from math import atan2, sqrt
import numpy as np


def calc_f(x, uin):
    return np.asarray([[uin[0, 0]], [uin[1, 0]], [uin[2, 0]], [0]], dtype=np.float64)


def calc_h(x, uin):
    x = x.flatten()
    u = x[0]
    v = x[1]
    w = x[2]
    c_alpha = x[3]

    a0 = atan2(w, u) * (1 + c_alpha)
    a1 = atan2(v, sqrt(u**2 + w**2))
    a2 = sqrt(u**2 + v**2 + w**2)

    h_mat = np.asarray([[a0], [a1], [a2]], dtype=np.float64)
    return h_mat


def calc_Fx(x, uin):
    return np.zeros((4, 4))


def calc_Hx(x, uin):
    u = x[0]
    v = x[1]
    w = x[2]
    c_alpha = x[3]

    V2 = u ** 2 + v ** 2 + w ** 2

    a00 = - (w / (u ** 2 + w ** 2)) * (1 + c_alpha)
    a01 = 0
    a02 = (u / (u ** 2 + w ** 2)) * (1 + c_alpha)
    a03 = atan2(w, u)
    a10 = - u * v / (V2 * sqrt(u ** 2 + w ** 2))
    a11 = sqrt(u ** 2 + w ** 2) / V2
    a12 = - v * w / (V2 * sqrt(u ** 2 + w ** 2))
    a13 = 0
    a20 = u / sqrt(V2)
    a21 = v / sqrt(V2)
    a22 = w / sqrt(V2)
    a23 = 0

    Hx_mat = np.asarray([[a00, a01, a02, a03],
                         [a10, a11, a12, a13],
                         [a20, a21, a22, a23]], dtype=np.float64)
    return Hx_mat
