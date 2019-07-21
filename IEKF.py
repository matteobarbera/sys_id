# from typing import Callable, List


# TODO Type checking is apparently bad practice in python, type hint instead?
# TODO add documentation
class MyIEKF:
    def __init__(self, n, calc_f, calc_h, calc_Fx, calc_Hx):
        # if not callable(calc_f) or not callable(calc_h) or not callable(calc_Fx) or not callable(calc_Hx):
        #     raise TypeError("Inputs must be function pointers")
        if n < 1:
            raise ValueError("Dimension of state vector should be greater than 0")

        self._calc_f = calc_f
        self._calc_h = calc_h
        self._calc_Fx = calc_Fx
        self._calc_Hx = calc_Hx

        self._n = n  # number of states
        self._N = None  # number of measurement samples
        self._dt = None  # time step discretization
        self._nm = None  # number of measurements (z vec)
        self._m = None  # number if inputs

        self._integration_routine = self._default_integrator

        # Default initial estimates
        self._Ex_0 = None  # initial optimal value x_k_1k_1
        self._x_0 = None  # initial state
        self._P_0 = None  # initial estimate for covariance matrix
        self._Ew = None  # initial noise bias

        self._Q = None  # system noise variance
        self._R = None  # measurement noise variance

    # TODO implement default numerical integration routine
    def _default_integrator(self):
        pass

    # TODO implement KF loop
    def _run(self):
        for k in range(self._N):
            pass

    def set_intial_conditions(self, x_0, Ex_0, P_0, Ew):
        self._x_0 = x_0
        self._Ex_0 = Ex_0
        self._P_0 = P_0
        self._Ew = Ew

    def set_variance_matrices(self, Q, R):
        self._Q = Q
        self._R = R

    def use_integrator(self, fn_integrator):
        if not callable(fn_integrator):
            raise TypeError("Input must be a function pointer")
        self._integration_routine = fn_integrator

    # TODO update name
    # TODO finish implementing user call of KF loop
    def dummy(self, Z_k, U_k, dt):
        # if type(Z_k) is not np.ndarray or type(U_k) is not np.ndarray:
        #     raise TypeError("Input arrays must be of type numpy ndarray")
        # if type(dt) is not float or type(dt) is not int:
        #     raise TypeError("dt should be an int or float")
        if Z_k.shape[1] != U_k.shape[1]:
            raise ValueError("The row length of the input arrays should be the same")
        if dt <= 0:
            raise ValueError("dt should be larger than 0")

        self._N = Z_k.shape[1]
        self._dt = dt
        self._nm = Z_k.shape[0]  # number of measurements
        self._m = U_k.shape[0]  # number of inputs

