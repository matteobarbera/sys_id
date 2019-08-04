from collections import defaultdict
from itertools import product

import numpy as np
from scipy import sparse
from scipy.linalg import inv
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

    def _cart2bary_data(self, targets):
        # find index of simplices target points are in
        idx_simplices = self.tri.find_simplex(targets)

        mask = idx_simplices != -1
        idx_simplices_valid = idx_simplices[mask]  # remove points not in any simplex

        # Affine transform defined by A b = x - v_0
        X = self.tri.transform[idx_simplices_valid, :self._dim]  # A^-1
        Y = targets[mask] - self.tri.transform[idx_simplices_valid, self._dim]  # x - v_0
        b = np.einsum('ijk,ik->ij', X, Y)
        b_coords = np.c_[b, 1 - b.sum(axis=-1)]

        for s_idx, data_b, data_y in zip(idx_simplices_valid, b_coords, self._y[mask]):
            self._simp_xb_dict[s_idx].append(data_b)
            self._simp_y_dict[s_idx].append(data_y)
            # TODO Store where each data point in cart coords is too?
            # B-Spline coefficients should be already correctly sorted with B-mat, prob not needed

    def cart2bary_simp(self, simp, target):
        # Barycentric coordinate calculation of point wrt simplex simp

        # Affine transform defined by A b = x - v_0
        diff_tv0 = (target - self.tri.transform[simp, self._dim]).T
        b = self.tri.transform[simp, :self._dim].dot(diff_tv0)
        b_coords = np.append(b, 1 - b.sum())
        return b_coords

    @staticmethod
    def multi_index_perm(d, n):
        """
        Computes all valid permutation of multi index k = (k_0, k_1, ..., k_n) with |k| = k_0 + k_1 + ... + k_n = d
        Returns a sorted array with descending multi index values ([[d, 0, 0], [d-1, 1, 0], ..., [0, 1, d-1], [0, 0, d]])
        :param d: int
        :param n: int
        :return: 2D ndarray
        """
        mi_perm = np.asarray(list(product(range(d + 1), repeat=n+1)))
        mi_perm_valid = mi_perm[mi_perm.sum(axis=-1) == d][::-1]
        return mi_perm_valid

    @staticmethod
    def b_spline_func(d, b_coords):
        k_permutations = SimplexSplines.multi_index_perm(d, b_coords.shape[-1] - 1)

        b_coords = np.expand_dims(b_coords, axis=1)  # Add axis for broadcasting
        b_block = np.power(b_coords, k_permutations).prod(axis=-1)

        # Calculate d! / k!
        b_coeffs = np.repeat(factorial(d), k_permutations.shape[0]) / factorial(k_permutations).prod(axis=1)
        b_block *= b_coeffs  # B-form polynomial
        return b_block

    def gen_b_rmat(self, d):
        self._cart2bary_data(self._x)

        b_blocks = []
        sorted_y = []
        for tj in range(self.tri.simplices.shape[0]):
            sorted_b_coords = np.asarray(self._simp_xb_dict[tj])[:, np.argsort(self.tri.simplices[tj])]
            b_blocks.append(sparse.coo_matrix(self.b_spline_func(d, sorted_b_coords)))
            sorted_y += self._simp_y_dict[tj]

        return sparse.block_diag(b_blocks, format='csr'), sorted_y

    def c_to_idx(self, c, s_idx, d):
        c_perm = SimplexSplines.multi_index_perm(d, self._dim)
        offset = c_perm.tolist().index(c.tolist())
        return s_idx * c_perm.shape[0] + offset

    def smoothness_mat(self, d, m):
        sorted_indices = np.argsort(self.tri.simplices)
        sorted_simp = np.take_along_axis(self.tri.simplices, sorted_indices, axis=-1)
        sorted_neigh = np.take_along_axis(self.tri.neighbors, sorted_indices, axis=-1)

        # Compute all possible permutations of LH part
        lhs_perm_dict = dict()
        perms = SimplexSplines.multi_index_perm(d, self._dim)
        for i in range(self._dim + 1):
            lhs_perm_dict[i] = perms[perms[:, i] == m]
        gamma_perm = SimplexSplines.multi_index_perm(m, self._dim)

        csr_mat_inp = []
        row = 0
        for s_idx, (simp, nbs) in enumerate(zip(sorted_simp, sorted_neigh)):
            for v_idx, (vtx, nb_idx) in enumerate(zip(simp, nbs)):
                if nb_idx > s_idx and nb_idx != -1:
                    lhs_perm = lhs_perm_dict[v_idx]

                    other_simp = sorted_simp[nb_idx]
                    other_vtx = np.setdiff1d(other_simp, simp)

                    vtx_bcoords = self.cart2bary_simp(s_idx, self.tri.points[other_vtx])
                    bspline_v = SimplexSplines.b_spline_func(m, vtx_bcoords)

                    for mi_l in lhs_perm:
                        csr_mat_inp.append([-1, row, self.c_to_idx(mi_l, s_idx, d)])

                        mi_r = np.insert(np.delete(mi_l, v_idx), other_simp.tolist().index(other_vtx), 0)
                        rhs_part = gamma_perm + mi_r
                        for rhs_part_coeff, b_v in zip(rhs_part, bspline_v):
                            csr_mat_inp.append([b_v, row, self.c_to_idx(rhs_part_coeff, nb_idx, d)])
                        row += 1
        csr_mat_inp = np.asarray(csr_mat_inp)
        return sparse.csr_matrix((csr_mat_inp[:, 0], (csr_mat_inp[:, 1], csr_mat_inp[:, 2])), shape=(row, (s_idx + 1) * perms.shape[0]))

    def _ols_debug(self, d, m):
        b_mat, y_vec = self.gen_b_rmat(d)
        h_mat = self.smoothness_mat(d, m)
        kkt_mat = sparse.bmat([[b_mat.transpose() * b_mat, h_mat.transpose()], [h_mat, None]])
        other_mat = np.append(b_mat.transpose() @ y_vec, np.zeros(h_mat.shape[0]))
        return inv(kkt_mat.todense()) @ other_mat


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
    ss._cart2bary_data(np.asarray([[0, -1], [0, -2 / 3]]))
    b_single = ss.cart2bary_simp(0, [0, -2/3])
    answ_Bty = [[0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]]
    assert((np.sort(ss._simp_xb_dict[0]) - answ_Bty).sum() < 1e-15)
    assert((b_single - answ_Bty[1]).sum() < 1e-15)

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
    answ_Bty = [-0.6, -0.2, -0.2, 1.4, 1.8, 1.8]
    Bt_Y = b_regmat.transpose() @ sorted_y
    assert((np.sort(Bt_Y) - np.sort(answ_Bty)).sum() < 1e-15)
    h_mat = ss2.smoothness_mat(1, 0).toarray()
    answ_hmat = [[0, -1, 0, 1, 0, 0], [0, 0, -1, 0, 0, 1]]
    assert((np.sort(h_mat) - np.sort(answ_hmat)).sum() < 1e-15)
    c_opt_inv = ss2._ols_debug(1, 0)
    answ_copt = [-2.5, 0, 2.5, 0, 2.5, 2.5, 0, 0]
    assert((np.sort(c_opt_inv) - np.sort(answ_copt)).sum() < 1e-15)
