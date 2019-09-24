from MyIEKF import MyIEKF
from kf_functions import calc_f, calc_h, calc_Fx, calc_Hx
import csv
import numpy as np
from matplotlib import pyplot as plt

with open('F16traindata_CMabV_2019.csv', 'r') as csv_f:
    csv_reader = csv.reader(csv_f, delimiter=',')

    Cm = []
    U_k = []
    Z_k = []
    for row in csv_reader:
        Cm.append(row[0])
        U_k.append(row[1:4])
        Z_k.append(row[4:])
    Cm = np.asarray(Cm, dtype=np.float64)
    U_k = np.asarray(U_k, dtype=np.float64)
    Z_k = np.asarray(Z_k, dtype=np.float64)

dt = 0.01
n = 4
G = np.eye(4)
# G[3, 3] = 0

x_0 = np.ones((4, 1))
Ex_0 = 100 * np.ones((4, 1))
Ex_0[3] = 0
P_0 = 100000 * np.eye(4)

Q = 1e-3 * np.eye(4)
Q[3, 3] = 0
R = np.zeros((3, 3))
R[0, 0] = 0.035
R[1, 1] = 0.013
R[2, 2] = 0.11

# TODO check rk4 SOMETHING WRONG HERE
# TODO check where w starts to blow up
IEKF = MyIEKF(n, calc_f, calc_h, calc_Fx, calc_Hx, G)
IEKF.set_intial_conditions(x_0, Ex_0, P_0)
IEKF.set_variance_matrices(Q, R)
x_filtered, stdx_filtered = IEKF.run_filter(Z_k, U_k, dt, iterate=False)

# print(x_filtered)
print(list(np.asarray(x_filtered).reshape(10001, 4)[:, 0]))
print(len(x_filtered))

plt.figure()
plt.plot(list(range(len(x_filtered))), np.asarray(x_filtered).reshape(10001, 4)[:, 0], label='u')
plt.legend()
plt.figure()
plt.plot(list(range(len(x_filtered))), np.asarray(x_filtered).reshape(10001, 4)[:, 1], label='v')
plt.legend()
plt.figure()
plt.plot(list(range(len(x_filtered))), np.asarray(x_filtered).reshape(10001, 4)[:, 2], label='w')
plt.legend()
plt.figure()
plt.plot(list(range(len(x_filtered))), np.asarray(x_filtered).reshape(10001, 4)[:, 3], label='Calpha')
plt.legend()
plt.show()
