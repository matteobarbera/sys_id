import numpy as np
from scipy.spatial import Delaunay


class SimplexSplines:
    def __init__(self, x):
        self._degree = min(x.shape)
        self._x = x

    def n_cube_domain(self, res):
        if res < 2:
            raise ValueError("Resolution must be at least 2")
        edge_points = []
        for n in range(self._degree):
            x_n = self._x[:, n]
            min_max = np.linspace(min(x_n), max(x_n), res)
            edge_points.append(min_max)
        mesh = np.meshgrid(*edge_points)
        points = np.dstack(mesh).reshape(res ** self._degree, self._degree)
        return points

    def triangulate(self, res=4):
        points = self.n_cube_domain(res)
        tri = Delaunay(points)
