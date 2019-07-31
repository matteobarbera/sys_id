import numpy as np
from scipy.spatial import Delaunay


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

        # Affine transform defined by A b = x - v_0
        X = (self.tri.transform[idx_simplices, :self._dim])  # A^-1
        Y = targets - self.tri.transform[idx_simplices, self._dim]  # x - v_0
        b = np.einsum('ijk,ik->ij', X, Y)
        b_coords = np.c_[b, 1 - b.sum(axis=1)]

        return b_coords


if __name__ == "__main__":
    x = np.asarray([list(range(10)), list(range(10))])
    # print(x.shape)
    ss = SimplexSplines(x)
    res = 4

    # points = ss._n_cube_domain(res)
    # print(points)
    # plt.plot(x[0], x[1], ls="None", marker="o")
    # plt.plot(points[:, 0], points[:, 1], ls="None", color='k', marker='x')

    t_points = [[1, 2], [0, 0], [2, 0]]
    ss.triangulate(custom_points=t_points, res=res)
    print(ss.cart2bary([[0.16, 0.30], [0.1, 0.6]]))
    print('-'*40)
    t_points = [[-1, -1], [1, -1], [0, 0]]
    ss.triangulate(custom_points=t_points, res=res)
    print(ss.cart2bary([[0, -1], [0, -2 / 3]]))
    # plt.triplot(ss.tri.points[:, 0], ss.tri.points[:, 1], ss.tri.simplices)

    # plt.show()
