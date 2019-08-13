import numpy as np
from numpy.linalg import inv

from my_c2d import my_c2d
from my_rk4 import my_rk4


# TODO add documentation
class MyIEKF:
    def __init__(self, n, calc_f, calc_h, calc_Fx, calc_Hx, G):
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

        self._integration_routine = my_rk4
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

        store_x = np.empty((self._n, self._N))
        store_stdx = np.empty((self._n, self._N))

        # TODO check if x_0 is actually needed
        x_kk = self._x_0  # x_kk used for estimation error calculation
        x_k_1k_1 = self._Ex_0
        P_k_1k_1 = self._P_0
        ti = 0
        tf = self._dt
        for k in range(self._N):
            # Prediction x(k+1|k)
            x_kk_1 = self._integration_routine(self._calc_f, x_k_1k_1, U_k[:, k], [ti, tf])
            x_kk = x_kk_1
            # Predicted output z(k+1|k)
            z_kk_1 = self._calc_h(x_kk_1, U_k[:, k])

            # Calculate Phi(k+1,k) and Gamma(k+1,k)
            Fx = self._calc_Fx(x_kk_1, U_k[:, k])
            Phi, Gamma = self._discretize_matrices(Fx, self._G, dt)

            # Prediction covariance matrix P(k+1|k)
            P_kk_1 = Phi @ P_k_1k_1 @ Phi.T + Gamma @ self._Q @ Gamma.T

            # TODO include iteration step
            if iterate:
                eta2 = x_kk_1
                epsilon = 1e-10
                err = 2 * epsilon
                max_iters = 100
                for i in range(max_iters):
                    if err < epsilon:
                        break
                    eta1 = eta2
                    Hx = self._calc_Hx(x_kk_1, U_k[:, k])

                    # Innovation matrix
                    Ve = Hx @ P_kk_1 @ Hx.T + self._R

                    # Kalman gain
                    K = P_kk_1 @ Hx.T @ inv(Ve)
                    # Observation state
                    z_p = self._calc_h(eta1, U_k[:, k])

                    eta2 = x_kk_1 + K @ (Z_k[:, k] - z_p - Hx @ (x_kk_1 - eta1))
                    err = np.linalg.norm(eta2 - eta1, np.inf) / np.linalg.norm(eta1, np.inf)
                else:
                    raise RuntimeError("Exceeded max IEKF iterations")
            else:
                Hx = self._calc_Hx(x_kk_1, U_k[:, k])
                # Covariance matrix of innovation
                Ve = Hx @ P_kk_1 @ Hx.T + self._R

                # Kalman gain K(k+1)
                K_gain = P_kk_1 @ Hx.T @ inv(Ve)
                # Calculate optimal state x(k+1|k+1)
                x_k_1k_1 = x_kk_1 + K_gain @ (Z_k[:, k] - z_kk_1)

            P_term = np.eye(self._n) - K_gain @ Hx
            P_k_1k_1 = P_term @ P_kk_1 @ P_term.T + K_gain @ self._R @ K_gain.T
            P_cor = np.diag(P_k_1k_1)
            stdx_cor = np.sqrt(P_cor)

            store_x[:, k] = x_k_1k_1
            store_stdx[:, k] = stdx_cor

            # Next step
            ti = tf
            tf += dt

            return store_x, store_stdx
