import numpy as np


def my_rk4(fn, xin, uin, t_vec):
    w = np.copy(xin)
    h = (t_vec[0] - t_vec[1]) / 2.

    for i in range(2):
        k1 = h * fn(w, uin)
        k2 = h * fn(w + k1 / 2., uin)
        k3 = h * fn(w + k2 / 2., uin)
        k4 = h * fn(w + k3, uin)

        w += (k1 + 2 * k2 + 2 * k3 + k4) / 6.

    return w
