import functools

import numpy as np
from numpy.linalg import inv


class MyLSPE:
    class _Decorators:
        @classmethod
        def update_a_mat(cls, func):
            @functools.wraps(func)
            def wrapper_update_a_mat(self, order=None):
                if order is None or order == self._order:
                    return func(self)
                else:
                    self._order = order
                    self._A = self._gen_a_poly()
                    return func(self)
            return wrapper_update_a_mat

    def __init__(self, x, y, order=1):
        self._N = len(x.flatten())
        self._x = x
        self._y = y.reshape(self._N, 1)
        self._order = order
        self._A = self._gen_a_poly()

    def _gen_a_poly(self):
        a_mat = np.empty((self._N, self._order + 1))
        a_mat[:, 0] = np.ones(self._N)
        for j in range(1, 1 + self._order):
            a_mat[:, j] = self._x ** j
        return a_mat

    @_Decorators.update_a_mat
    def ols(self, order=None):
        return np.matmul(np.matmul(inv(np.matmul(self._A.T, self._A)), self._A.T), self._y)


if __name__ == "__main__":
    x = np.asarray(list(range(10)))
    y = x

    PE = MyLSPE(x, y)
    PE.ols()
    print(PE._A)
    PE.ols()
    print(PE._A)



