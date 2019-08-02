from collections import defaultdict
from itertools import product

import numpy as np
from scipy.sparse import coo_matrix, block_diag
from scipy.spatial import Delaunay
from scipy.special import factorial


class SimplexSplines:
    def __init__(self, x, y):
        self._dim = min(x.shape)
        self._x = x.reshape(tuple(sorted(x.shape)[::-1]))  # column major order
        self._y = y
        self.tri = None

        self._simp_xb_dict = defaultdict(list)
        self._simp_y_dict = defaultdict(list)

    def _n_cube_domain(self, res):
        if res < 2:
            raise ValueError("Resolution must be at least 2")
        edge_points = []
        for n in range(self._dim):
            x_n = self._x[:, n]
            min_max = np.linspace(min(x_n), max(x_n), res)
            edge_points.append(min_max)
        mesh = np.meshgrid(*edge_points)
        points = np.dstack(mesh).reshape(res ** self._dim, self._dim)  # reshape into 2D array of (x, y, ...) for Delaunay
        return points

    def triangulate(self, *, custom_points=None, res=4):
        if custom_points is None:
            points = self._n_cube_domain(res)
        else:
            points = custom_points
        self.tri = Delaunay(points)

    def _cart2bary(self, targets):
        # find index of simplices target points are in
        idx_simplices = self.tri.find_simplex(targets)

        mask = idx_simplices != -1
        idx_simplices_valid = idx_simplices[mask]  # remove points not in any simplex

        # Affine transform defined by A b = x - v_0
        X = (self.tri.transform[idx_simplices_valid, :self._dim])  # A^-1
        Y = targets[mask] - self.tri.transform[idx_simplices_valid, self._dim]  # x - v_0
        b = np.einsum('ijk,ik->ij', X, Y)
        b_coords = np.c_[b, 1 - b.sum(axis=1)]

        for s_idx, data_b, data_y in zip(idx_simplices_valid, b_coords, self._y[mask]):
            self._simp_xb_dict[s_idx].append(data_b)
            self._simp_y_dict[s_idx].append(data_y)
            # TODO Store where each data point in cart coords is too?
            # B-Spline coefficients should be already correctly sorted with B-mat, prob not needed

    @staticmethod
    def multi_index_perm(d, n):
        """
        Computes all valid permutation of multi index k = (k_0, ..., k_n) with |k| = d
        Returns a sorted array with descending multi index combinations ([[d, 0, 0], ..., [0, 0, d]])
        :param d: int
        :param n: int
        :return: 2D ndarray
        """
        mi_perm = np.asarray(list(product(range(d + 1), repeat=n)))
        mi_perm_valid = mi_perm[mi_perm.sum(axis=-1) == d][::-1]
        return mi_perm_valid

    @staticmethod
    def b_spline_func(d, b_coords):
        # Compute only relevant permutations of k (|k| = degree)
        k_permutations = SimplexSplines.multi_index_perm(d, b_coords.shape[-1])

        b_coords = np.expand_dims(b_coords, axis=1)  # Add axis for broadcasting
        b_block = np.power(b_coords, k_permutations).prod(axis=-1)

        # Calculate d! / k!
        b_coeffs = np.repeat(factorial(d), k_permutations.shape[0]) / factorial(k_permutations).prod(axis=1)
        b_block *= b_coeffs  # B-form polynomial
        return b_block

    def gen_b_rmat(self, d):
        self._cart2bary(self._x)

        b_blocks = []
        sorted_y = []
        for tj in range(self.tri.simplices.shape[0]):
            sorted_b_coords = np.asarray(self._simp_xb_dict[tj])[:, np.argsort(self.tri.simplices[tj])]
            b_blocks.append(coo_matrix(self.b_spline_func(d, sorted_b_coords)))
            sorted_y += self._simp_y_dict[tj]

        return block_diag(b_blocks), sorted_y


if __name__ == "__main__":
    x = np.random.random(1000).reshape(2, 500)
    y = np.random.random(500)
    ss = SimplexSplines(x, y)
    res = 3

    # import matplotlib.pyplot as plt
    # plt.plot(x[0], x[1], ls="None", marker="o")
    # points = ss._n_cube_domain(res)
    # plt.plot(points[:, 0], points[:, 1], ls="None", color='k', marker='x')
    # ss.triangulate(res=res)
    # plt.triplot(ss.tri.points[:, 0], ss.tri.points[:, 1], ss.tri.simplices)
    # plt.show()
    # print(ss.tri.neighbors)

    # Exercise 1 L06 Barycentric coordinate transformation
    t_points = [[0, 0], [1, -1], [-1, -1]]
    ss._y = np.asarray([1, 1])  # needed to make test work
    ss.triangulate(custom_points=t_points, res=res)
    ss._cart2bary(np.asarray([[0, -1], [0, -2 / 3]]))
    answ = [[0, 0.5, 0.5], [1/3, 1/3, 1/3]]
    assert((np.sort(ss._simp_xb_dict[0]) - answ).sum() < 1e-15)

    # Exercise 2 L06 Simplex polynomial evaluation
    b_coords = np.array([[0.5, 0.25, 0.25]])
    ex2 = SimplexSplines.b_spline_func(2, b_coords) * [2, -4, 4, -4, 3, 2]
    assert (ex2.sum() == 0.75)

    # Example 6.8 L06 Estimating a linear simplex spline
    x2 = np.asarray([[0.2, 0.8], [0.4, 0.2], [0.8, 0.2], [0.8, 0.6]])
    y2 = np.asarray([-1, 2, 2, 1])
    ss2 = SimplexSplines(x2, y2)
    ss2.triangulate(custom_points=[[0, 1], [0, 0], [1, 0], [1, 1]])
    b_regmat, sorted_y = ss2.gen_b_rmat(1)
    answ = [-0.6, -0.2, -0.2, 1.4, 1.8, 1.8]
    Bt_Y = b_regmat.T @ sorted_y
    assert((np.sort(Bt_Y) - np.sort(answ)).sum() < 1e-15)
