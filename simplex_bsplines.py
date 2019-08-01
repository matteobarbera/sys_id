from itertools import product

import numpy as np
from scipy.spatial import Delaunay
from scipy.special import factorial


class SimplexSplines:
    def __init__(self, x):
        self._dim = min(x.shape)
        self._x = x.reshape(tuple(sorted(x.shape)))
        self.tri = None

    def _n_cube_domain(self, res):
        if res < 2:
            raise ValueError("Resolution must be at least 2")
        edge_points = []
        for n in range(self._dim):
            x_n = self._x[n, :]
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

    def cart2bary(self, targets):
        # find index of simplices target points are in
        idx_simplices = self.tri.find_simplex(targets)
        idx_simplices_valid = idx_simplices[idx_simplices != -1]  # remove points not in any simplex

        # Affine transform defined by A b = x - v_0
        X = (self.tri.transform[idx_simplices_valid, :self._dim])  # A^-1
        Y = targets - self.tri.transform[idx_simplices_valid, self._dim]  # x - v_0
        b = np.einsum('ijk,ik->ij', X, Y)
        b_coords = np.c_[b, 1 - b.sum(axis=1)]

        return b_coords

    @staticmethod
    def b_spline_func(d, b_coords):
        # Compute only relevant permutations of k (|k| = degree)
        k_permutations = np.asarray(list(product(range(d + 1), repeat=b_coords.shape[-1])))
        k_permutations = k_permutations[k_permutations.sum(axis=-1) == d][::-1]

        b_coords = np.expand_dims(b_coords, axis=1)  # Add axis for broadcasting
        b_block = np.power(b_coords, k_permutations).prod(axis=-1)

        # Calculate d! / k!
        b_coeffs = np.repeat(factorial(d), k_permutations.shape[0]) / factorial(k_permutations).prod(axis=1)
        b_block *= b_coeffs  # B-form polynomial
        return b_block


if __name__ == "__main__":
    x = np.random.random(1000).reshape(2, 500)
    ss = SimplexSplines(x)
    res = 4

    # import matplotlib.pyplot as plt
    # plt.plot(x[0], x[1], ls="None", marker="o")
    # points = ss._n_cube_domain(res)
    # plt.plot(points[:, 0], points[:, 1], ls="None", color='k', marker='x')
    # ss.triangulate(res=res)
    # print(ss.tri.simplices)
    # plt.triplot(ss.tri.points[:, 0], ss.tri.points[:, 1], ss.tri.simplices)
    # plt.show()

    # Exercise 1 L06 Barycentric coordinate transformation
    t_points = [[1, -1], [0, 0], [-1, -1]]
    ss.triangulate(custom_points=t_points, res=res)
    b_coords = ss.cart2bary([[0, -1], [0, -2 / 3]])
    answ = [[0, 0.5, 0.5], [1/3, 1/3, 1/3]]
    assert((np.sort(b_coords) - answ).sum() < 1e-15)

    # Exercise 2 L06 Simplex polynomial evaluation
    b_coords = np.array([[0.5, 0.25, 0.25]])
    ex2 = SimplexSplines.b_spline_func(2, b_coords) * [2, -4, 4, -4, 3, 2]
    assert (ex2.sum() == 0.75)
