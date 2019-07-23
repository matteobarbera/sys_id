import numpy as np
from numpy.linalg import inv
from scipy.integrate import odeint

from my_c2d import my_c2d


# TODO Type checking is apparently bad practice in python, type hint instead?
# TODO add documentation
class MyIEKF:
    def __init__(self, n, calc_f, calc_h, calc_Fx, calc_Hx, G):
        # if not callable(calc_f) or not callable(calc_h) or not callable(calc_Fx) or not callable(calc_Hx):
        #     raise TypeError("Inputs must be function pointers")
        if n < 1:
            raise ValueError("Dimension of state vector should be greater than 0")

        self._calc_f = calc_f
        self._calc_h = calc_h
        self._calc_Fx = calc_Fx
        self._calc_Hx = calc_Hx

        self._G = G

        self._n = n  # number of states
        self._N = None  # number of measurement samples
        self._dt = None  # time step discretization
        self._nm = None  # number of measurements (z vec)
        self._m = None  # number if inputs

        # FIXME odeint only accepts functions of form f(y,t)
        self._integration_routine = odeint
        self._discretize_matrices = my_c2d

        # Default initial estimates
        self._Ex_0 = None  # initial optimal value x_k_1k_1
        self._x_0 = None  # initial state
        self._P_0 = None  # initial estimate for covariance matrix
        self._Ew = None  # initial noise bias

        self._Q = None  # system noise variance
        self._R = None  # measurement noise variance

    def set_intial_conditions(self, x_0, Ex_0, P_0, Ew):
        self._x_0 = x_0
        self._Ex_0 = Ex_0
        self._P_0 = P_0
        self._Ew = Ew

    def set_variance_matrices(self, Q, R):
        self._Q = Q
        self._R = R

    def run_filter(self, Z_k, U_k, dt, iterate=True):
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

        if self._x_0 is None:
            raise RuntimeError("Initial conditions need to be specified (use set_initial_conditions())")
        if self._Q is None:
            raise RuntimeError("Variance matrices Q and R need to be specified (use set_variance_matrices()")

        x_kk = self._x_0
        x_k_1k_1 = self._Ex_0
        P_k_1k_1 = self._P_0
        ti = 0
        tf = self._dt
        # TODO pass parameters to the calc functions
        for k in range(self._N):
            # Prediction x(k+1|k)
            x_kk_1 = self._integration_routine(self._calc_f, x_kk, np.linspace(ti, tf, num=10))
            x_kk = x_kk_1
            # Predicted output z(k+1|k)
            z_kk_1 = self._calc_h()

            # Calculate Phi(k+1,k) and Gamma(k+1,k)
            Fx = self._calc_Fx()
            Phi, Gamma = self._discretize_matrices(Fx, self._G, dt)

            # Prediction covariance matrix P(k+1|k)
            P_kk_1 = np.matmul(np.matmul(Phi, P_k_1k_1), Phi.T) + \
                     np.matmul(np.matmul(Gamma, self._Q), Gamma.T)

            # TODO include iteration step
            if iterate:
                raise NotImplemented("Iteration step of IEKF not implemented!")
            else:
                Hx = self._calc_Hx()
                # Covariance matrix of innovation
                Ve = np.matmul(np.matmul(Hx, P_kk_1), Hx.T) + self._R

                # Kalman gain K(k+1)
                K_gain = np.matmul(np.matmul(P_kk_1, Hx.T), inv(Ve))
                # Calculate optimal state x(k+1|k+1)
                x_k_1k_1 = x_kk_1 + np.matmul(K_gain, (Z_k[:, k] - z_kk_1))

            P_term = np.eye(self._n) - np.matmul(K_gain, Hx)
            P_k_1k_1 = np.matmul(np.matmul(P_term, P_kk_1), P_term.T) + \
                       np.matmul(np.matmul(K_gain, self._R), K_gain.T)

            # Next step
            ti = tf
            tf += dt

            # TODO make store result function, eventually let user specify how much to store
