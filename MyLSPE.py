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
    print(f"{' Running MyLSPE.py ':=^40}")
    print("Performing unit tests...")
    # OLS Example from AE4320 lecture 4
    x_ols = [0, 1, 2, 3]
    y_ols = [-1, 2, 5, 1]
    PE_test = MyLSPE(x_ols, y_ols, order=2)
    theta_ols = PE_test.ols()
    theta_ols_matlab = np.asarray([[-1.3500], [6.1500], [-1.7500]])
    assert (np.sum(np.round(theta_ols, 4) - theta_ols_matlab) == 0)

    # WLS Example from AE4320 lecture 4
    var_wls = np.asarray([0.1, 0.1, 1, 0.1]) ** 2
    theta_wls = PE_test.wls(var_wls)
    theta_wls_matlab = np.asarray([[-1.0077], [4.2102], [-1.1795]])
    assert(np.sum(np.round(theta_wls, 4) - theta_wls_matlab) == 0)

    print("Unit tests passed!")
