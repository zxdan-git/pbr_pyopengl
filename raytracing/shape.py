from abc import ABC, abstractmethod
import numpy as np
from .ray import Ray, RayIntersectObject
from .bounding_box import AABB
from .util import det3x3
from enum import IntFlag


class Shape(RayIntersectObject):
    class PaintMode(IntFlag):
        FACE = 1
        LINE = 2
        FACE_AND_LINE = FACE | LINE

    def __init__(self):
        self._vertex = np.empty((0, 3))
        self._face_index = np.array([], dtype=np.uint32)
        self._line_index = np.array([], dtype=np.uint32)
        self._transform = np.identity(4)
        self._inv_transform = np.identity(4)
        self.paint_mode = self.PaintMode.FACE
        self._bbx = AABB()

    @property
    def vertex(self):
        return self._vertex

    @property
    def face_index(self):
        return self._face_index

    @property
    def line_index(self):
        return self._line_index

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, new_transform):
        if np.array_equal(new_transform, self._transform):
            return
        self._transform = new_transform
        self._inv_transform = np.linalg.inv(new_transform)
        self._update_bounding_box()

    @property
    def bounding_box(self):
        return self._bbx

    def centroid(self):
        return self._bbx.center()

    def ray_intersect(self, ray: Ray):
        # Intersection of a shape would update the t_max of ray.
        intersect = self._ray_intersect(ray)
        if intersect is None or intersect > ray.t_max:
            return None
        ray.t_max = intersect
        return intersect

    def ray_intersect_cost(self):
        return 1

    def _update_bounding_box(self):
        self._bbx = AABB()
        for i in range(self._vertex.shape[0]):
            t_pos = self._transform @ np.append(self._vertex[i], 1)
            self._bbx.embrace(t_pos[:3])

    @abstractmethod
    def _ray_intersect(self, ray: Ray):
        return None


class Sphere(Shape):
    def __init__(self, nu, nv):
        super().__init__()
        self.__generate_vertex(nu, nv)
        self.__generate_face_index(nu, nv)
        self.__generate_line_index(nu, nv)
        self._bbx = AABB(-1, 1, -1, 1, -1, 1)

    def _ray_intersect(self, ray: Ray):
        """
        suppose the ray intersect with the sphere at o + t.d

        ||o + t.d|| = 1

        ||d||^2t^2 + 2.o.d.t + ||o||^2 - 1 = 0

        t = -2.o.d +/- sqrt(4(o.d)^2 - 4||d||^2(||o||^2 - 1)) / (2||d||^2)
        """
        t_ray = Ray.transform(ray, self._inv_transform)
        term_a = np.power(np.linalg.norm(t_ray.dir), 2)
        if np.isclose(term_a, 0):
            return None
        term_b = 2 * np.dot(t_ray.dir, t_ray.pos)
        term_c = np.power(np.linalg.norm(t_ray.pos), 2) - 1

        discriminant = term_b * term_b - 4 * term_a * term_c
        if discriminant < 0:
            return None

        t_1 = (-term_b + np.sqrt(discriminant)) / (2 * term_a)
        t_2 = (-term_b - np.sqrt(discriminant)) / (2 * term_a)
        return np.min([t_1, t_2])

    def __generate_vertex(self, nu, nv):
        vertex = []
        for i in range(nu + 1):
            theta = np.pi * float(i) / nu
            y = np.cos(theta)
            r = np.sin(theta)
            for j in range(nv + 1):
                phi = 2 * np.pi * float(j) / nv
                vertex.append(
                    np.array([r * np.sin(phi), y, r * np.cos(phi)], dtype=np.float32)
                )
        self._vertex = np.array(vertex)

    def __generate_face_index(self, nu, nv):
        index = []
        n_col = nv + 1
        for i in range(nu):
            for j in range(nv):
                index += [
                    i * n_col + j,
                    (i + 1) * n_col + j,
                    (i + 1) * n_col + j + 1,
                    i * n_col + j,
                    (i + 1) * n_col + j + 1,
                    i * n_col + j + 1,
                ]
        self._face_index = np.array(index, dtype=np.uint32)

    def __generate_line_index(self, nu, nv):
        index = []
        n_col = nv + 1
        for i in range(nu):
            for j in range(nv):
                index += [
                    i * n_col + j,
                    i * n_col + j + 1,
                    i * n_col + j,
                    (i + 1) * n_col + j,
                ]
        self._line_index = np.array(index, dtype=np.uint32)


class Cube(Shape):
    def __init__(self):
        super().__init__()
        self._vertex = np.array(
            [
                # top points
                [-1, 1, 1],
                [1, 1, 1],
                [1, 1, -1],
                [-1, 1, -1],
                # bottom points
                [-1, -1, 1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=np.float32,
        )

        self._face_index = np.array(
            [
                # top
                0,
                1,
                2,
                0,
                2,
                3,
                # bottom
                4,
                6,
                5,
                4,
                7,
                6,
                # left
                0,
                3,
                4,
                4,
                3,
                7,
                # right
                1,
                5,
                2,
                2,
                5,
                6,
                # front
                0,
                5,
                1,
                0,
                5,
                5,
                # back
                3,
                2,
                6,
                3,
                6,
                7,
            ],
            dtype=np.uint32,
        )

        self._line_index = np.array(
            [
                # top
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                0,
                # bottom
                4,
                5,
                5,
                6,
                6,
                7,
                7,
                4,
                # left
                0,
                4,
                4,
                7,
                7,
                3,
                3,
                0,
                # right
                1,
                5,
                5,
                6,
                6,
                2,
                2,
                1,
                # front
                0,
                1,
                1,
                5,
                5,
                4,
                4,
                0,
                # back
                3,
                2,
                2,
                6,
                6,
                7,
                7,
                3,
            ],
            dtype=np.uint32,
        )
        self._bbx = AABB(-1, 1, -1, 1, -1, 1)

    def _ray_intersect(self, ray: Ray):
        t_ray = Ray.transform(ray, self._inv_transform)
        bbx = AABB(-1, 1, -1, 1, -1, 1)
        return bbx.ray_intersect(t_ray)


class Triangle(Shape):
    def __init__(self, v0, v1, v2):
        super().__init__()
        self._vertex = np.array([v0, v1, v2], dtype=np.float32)
        self._face_index = np.array([0, 1, 2], dtype=np.uint32)
        self._line_index = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint32)
        for v in [v0, v1, v2]:
            self._bbx.embrace(v)

    def _ray_intersect(self, ray: Ray):
        """
        alpha (v1 - v0) + beta (v2 - v0) + v0 = o + t.d
        (v1 - v0, v2 - v0, -d) @ (alpha, beta, t) = o - v0
        alpha, beta, t = inv((v1 - v0, v2 - v0, -d)) @ (o - v0)
        """
        t_ray = Ray.transform(ray, self._inv_transform)
        v0, v1, v2 = self._vertex
        cofficient = np.array([v1 - v0, v2 - v0, -t_ray.dir], dtype=np.float32)
        b = t_ray.pos - v0
        det = det3x3(cofficient)
        if np.isclose(det, 0):
            return None

        inv_det = 1 / det

        alpha = det3x3(np.array([b, v2 - v0, -t_ray.dir], dtype=np.float32)) * inv_det
        if alpha < 0 or alpha > 1:
            return None

        beta = det3x3(np.array([v1 - v0, b, -t_ray.dir], dtype=np.float32)) * inv_det
        if beta < 0 or beta > 1:
            return None

        gamma = alpha + beta
        if gamma < 0 or gamma > 1:
            return None

        t = det3x3(np.array([v1 - v0, v2 - v0, b], dtype=np.float32)) * inv_det
        if t < 0:
            return None
        return t
