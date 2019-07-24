import functools

import numpy as np
from numpy.linalg import inv


# TODO add documentation
class MyLSPE:
    class _Decorators:
        @classmethod
        def update_a_mat(cls, func):
            @functools.wraps(func)
            def wrapper_update_a_mat(self, *args, **kwargs):
                try:
                    if kwargs['order'] == self._order:
                        return func(self, *args, **kwargs)
                    else:
                        self._order = kwargs['order']
                        self._gen_a_poly()
                        return func(self, *args, **kwargs)
                except KeyError:
                    return func(self, *args, **kwargs)

            return wrapper_update_a_mat

    def __init__(self, x, y, order=1):
        if len(x) != len(y):
            raise ValueError("State and measurement data arrays need to be of equal length")
        self._N = len(x)
        self._x = np.asarray(x)
        self._y = np.asarray(y).reshape(self._N, 1)
        self._order = order
        self._gen_a_poly()

    def _gen_a_poly(self):
        a_mat = np.empty((self._N, self._order + 1))
        a_mat[:, 0] = np.ones(self._N)
        for j in range(1, 1 + self._order):
            a_mat[:, j] = self._x ** j
        self._A = a_mat

    @_Decorators.update_a_mat
    def ols(self, order=None):
        return inv(self._A.T @ self._A) @ self._A.T @ self._y

    @_Decorators.update_a_mat
    def wls(self, w_diag, order=None):
        w_mat = np.diag(w_diag)
        return inv(self._A.T @ inv(w_mat) @ self._A) @ self._A.T @ inv(w_mat) @ self._y


if __name__ == "__main__":
    # TODO add unit tests
    x = list(range(10))
    y = x
    PE = MyLSPE(x, y)
    print(PE.ols())
    print(PE.ols(order=3))
    w_diag = np.ones(10)
    print(PE.wls(w_diag))
    print(PE.wls(w_diag, order=2))
