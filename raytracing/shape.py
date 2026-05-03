from abc import abstractmethod
from enum import IntFlag
from typing import Tuple

import numpy as np

from .bounding_box import AABB
from .constants import INF, zero2f, zero3f, ZERO3F
from .intersection import Intersection
from .material import Material
from .ray import Ray
from .ray_intersect_object import RayIntersectObject
from .transform import transform_dir, transform_pos
from .shape_sample_util import (
    uniform_sample_sphere,
    uniform_sphere_pdf,
    uniform_sample_triangle,
    uniform_sample_triangle_pdf,
)
from .util import det3x3, normalize
from .typing import Vec3f, Vec2f


class Shape(RayIntersectObject):
    class PaintMode(IntFlag):
        FACE = 1
        LINE = 2
        FACE_AND_LINE = FACE | LINE

    def __init__(self):
        self._vertex = np.empty((0, 3))
        self.__transformed_vertex = np.empty((0, 3))
        self._face_index = np.array([], dtype=np.uint32)
        self._line_index = np.array([], dtype=np.uint32)
        self._transform = np.identity(4)
        self._inv_transform = np.identity(4)
        self.paint_mode = self.PaintMode.FACE
        self._bbx = AABB()
        self._area = -1
        self.material: Material = None

    @property
    def vertex(self):
        return self._vertex

    @property
    def transformed_vertex(self):
        if self.__transformed_vertex.shape[0] == 0:
            self.__transformed_vertex = self._vertex.copy()
        return self.__transformed_vertex

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
        self.__transformed_vertex = np.array(
            [transform_pos(self.transform, v) for v in self._vertex], dtype=np.float32
        )
        self._update_bounding_box()
        self._update_area()

    @property
    def bounding_box(self):
        if self._bbx.empty():
            self._update_bounding_box()
        return self._bbx

    @property
    def area(self):
        if self._area == -1:
            self._update_area()
        return self._area

    def ray_intersect(self, ray: Ray) -> Intersection:
        # Intersection of a shape would update the t_max of ray.
        t, intersection = self._ray_intersect(ray)
        if intersection is None or t > ray.t_max:
            return None
        ray.t_max = t
        return intersection

    def ray_intersect_cost(self):
        return 1

    @abstractmethod
    def sample(self, u: Vec2f) -> Intersection:
        return Intersection()

    def sample_for_target(self, target: Vec3f, u: Vec2f) -> Intersection:
        return self.sample(u)

    def sample_pdf(self, intersection: Intersection):
        return 1 / self.area()

    def sample_pdf_for_target(self, target: Vec3f, dir: Vec3f):
        dir_norm = normalize(dir)
        ray = Ray(target, dir_norm)
        intersection = self.ray_intersect(ray)
        if intersection is None:
            return 0
        dist = intersection.pos - target
        dist_len = np.linalg.norm(dist)
        """
        p(A) = 1 / area.
        dw = dA * cos(theta) / r^2.
        P(w) = P(A(w))
        p(w) = p(A) * dA / dw
        p(w) = p(A) * r^2 / cos(theta) = r^2 / (area * cos(theta))
        """
        return dist_len * dist_len / self.area / np.abs(np.dot(-dir, intersection.n))

    def normal_to_world(self, n_local) -> Vec3f:
        n_world = transform_dir(np.transpose(self._inv_transform), n_local)
        return normalize(n_world)

    def pos_to_world(self, pos_local) -> Vec3f:
        return transform_pos(self._transform, pos_local)

    def _update_bounding_box(self):
        self._bbx = AABB()
        vertex = self.transformed_vertex
        for v_t in vertex:
            self._bbx.embrace(v_t)

    def _update_area(self):
        self._area = 0
        vertex = self.transformed_vertex
        for i in range(0, self._face_index.shape[0], 3):
            p0 = vertex[self._face_index[i]]
            p1 = vertex[self._face_index[i + 1]]
            p2 = vertex[self._face_index[i + 2]]
            self._area += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))

    @abstractmethod
    def _ray_intersect(self, ray: Ray) -> Tuple[np.float32, Intersection]:
        return INF, None


class Sphere(Shape):
    def __init__(self, nu, nv):
        super().__init__()
        self._generate_vertex(nu, nv)
        self._generate_face_index(nu, nv)
        self._generate_line_index(nu, nv)
        self._bbx = AABB(-1, 1, -1, 1, -1, 1)

    def sample(self, u: Vec2f) -> Intersection:
        intersection = Intersection()
        local_pos = uniform_sample_sphere(u)
        intersection.pos = self.pos_to_world(local_pos)
        intersection.n = self.normal_to_world(normalize(local_pos))
        intersection.uv = self._get_uv_for_local_pos(local_pos)
        intersection.sample_pdf = uniform_sphere_pdf()
        intersection.mat = self.material
        return intersection

    """
    def sample_for_target(self, target: Vec3f, u: Vec2f) -> Intersection:
        return None
    """

    def _ray_intersect(self, ray: Ray) -> Tuple[np.float32, Intersection]:
        """
        suppose the ray intersect with the sphere at o + t.d

        ||o + t.d|| = 1

        ||d||^2t^2 + 2.o.d.t + ||o||^2 - 1 = 0

        t = [-2.o.d +/- sqrt(4(o.d)^2 - 4||d||^2(||o||^2 - 1))] / (2||d||^2)
        """
        ray_t = Ray.transform(ray, self._inv_transform)
        term_a = np.power(np.linalg.norm(ray_t.dir), 2)
        if np.isclose(term_a, 0):
            return INF, None
        term_b = 2 * np.dot(ray_t.dir, ray_t.pos)
        term_c = np.power(np.linalg.norm(ray_t.pos), 2) - 1

        discriminant = term_b * term_b - 4 * term_a * term_c
        if discriminant < 0:
            return INF, None

        t_1 = (-term_b + np.sqrt(discriminant)) / (2 * term_a)
        t_2 = (-term_b - np.sqrt(discriminant)) / (2 * term_a)
        t = INF
        if t_2 >= 0:
            t = t_2
        elif t_1 >= 0:
            t = t_1
        else:
            return t, None

        pos_t = ray_t.at(t)
        n = self.normal_to_world(pos_t)
        uv = self._get_uv_for_local_pos(pos_t)
        return t, Intersection(ray.at(t), n, uv, self.material)

    def _get_uv_for_local_pos(self, pos_t: Vec3f) -> Vec2f:
        theta = np.acos(pos_t[1])
        phi = np.atan2(pos_t[0], pos_t[2])
        return np.array([0.5 + 0.5 * phi / np.pi, theta / np.pi], dtype=np.float32)

    def _generate_vertex(self, nu, nv):
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

    def _generate_face_index(self, nu, nv):
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

    def _generate_line_index(self, nu, nv):
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
                4,
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

    def sample(self, u: Vec2f) -> Intersection:
        intersection = Intersection()
        """
        Select a face with the first random number u0.
        0: x = 1
        1: x = -1
        2: y = 1
        3: y = -1
        4: z = 1
        5: z = -1
        """
        local_pos = zero3f()
        local_n = zero3f()
        face_idx = int(6 * u[0]) % 6
        axis = face_idx // 2
        if face_idx % 2 == 0:
            local_pos[axis] = 1
            local_n[axis] = 1
        else:
            local_pos[axis] = -1
            local_n[axis] = -1

        """
        Remap u0 and use u0 and u1 to sample the other two axis.
        """
        remapped_u0 = 6 * u[0] - face_idx
        local_pos[(axis + 1) % 3] = -1 + 2 * remapped_u0
        local_pos[(axis + 2) % 3] = -1 + 2 * u[1]

        intersection.pos = self.pos_to_world(local_pos)
        intersection.n = self.normal_to_world(local_n)
        intersection.uv = np.array([remapped_u0, u[1]], dtype=np.float32)
        intersection.sample_pdf = 1 / self.area()
        intersection.mat = self.material
        return intersection

    def _ray_intersect(self, ray: Ray) -> Tuple[np.float32, Intersection]:
        ray_t = Ray.transform(ray, self._inv_transform)
        bbx = AABB(-1, 1, -1, 1, -1, 1)
        t = bbx.ray_intersect(ray_t)
        if t is None:
            return INF, None

        pos = ray.at(t)
        pos_t = ray_t.at(t)
        n_t = zero3f()
        uv = zero2f()
        for i in range(3):
            for dir in [1, -1]:
                if np.isclose(pos_t[i], dir):
                    n_t[i] = dir
                    uv[0] = 0.5 + 0.5 * pos_t[(i + 1) % 3]
                    uv[1] = 0.5 + 0.5 * pos_t[(i + 2) % 3]
                    break
        if np.allclose(n_t, ZERO3F):
            raise ValueError(str(pos_t) + str(n_t) + "the normal is zero")
        n = self.normal_to_world(n_t)
        return t, Intersection(pos, n, uv, self.material)


class Triangle(Shape):
    def __init__(
        self,
        v0: Vec3f,
        v1: Vec3f,
        v2: Vec3f,
        uv0: Vec2f = None,
        uv1: Vec2f = None,
        uv2: Vec2f = None,
    ):
        super().__init__()
        self._vertex = np.array([v0, v1, v2], dtype=np.float32)
        if (not uv0 is None) and (not uv1 is None) and (not uv2 is None):
            self._tex_coord = np.array([uv0, uv1, uv2], dtype=np.float32)
        else:
            self._tex_coord = self._get_default_tex_coord()
        self._face_index = np.array([0, 1, 2], dtype=np.uint32)
        self._line_index = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint32)
        for v in [v0, v1, v2]:
            self._bbx.embrace(v)

    def sample(self, u: Vec2f) -> Intersection:
        intersection = Intersection()
        vertex = self.transformed_vertex
        intersection.pos = uniform_sample_triangle(u, vertex)
        intersection.n = normalize(
            np.cross(
                vertex[1] - vertex[0],
                vertex[2] - vertex[0],
            )
        )
        alpha, beta = self._get_barycentric_coord(intersection.pos)
        intersection.uv = np.transpose(self._tex_coord) @ np.array(
            [1 - alpha - beta, alpha, beta]
        )
        intersection.sample_pdf = 1 / self.area
        intersection.mat = self.material
        return intersection

    def _get_barycentric_coord(self, pos: Vec3f) -> Vec2f:
        vertex = self.transformed_vertex
        area_1 = 0.5 * np.linalg.norm(
            np.cross(
                pos - vertex[0],
                vertex[2] - vertex[0],
            )
        )
        alpha = area_1 / self.area

        area_2 = 0.5 * np.linalg.norm(
            np.cross(
                vertex[1] - vertex[0],
                pos - vertex[0],
            )
        )
        beta = area_2 / self.area
        return np.array([alpha, beta], dtype=np.float32)

    def _get_default_tex_coord(self):
        """
        Select the longest side whose ends are (0, 0) and (0, 1) and build a
        texture coordinate.
        """
        max_len = -1
        idx = -1
        for i in range(3):
            v = self._vertex[(i + 1) % 3] - self._vertex[i]
            v_len = np.linalg.norm(v)
            if max_len < v_len:
                max_len = v_len
                idx = i
        tex_coord = np.zeros((3, 2), dtype=np.float32)
        tex_coord[idx] = np.array([0, 0], dtype=np.float32)
        tex_coord[(idx + 1) % 3] = np.array([0, 1], dtype=np.float32)

        v1 = self._vertex[(idx + 1) % 3] - self._vertex[idx]
        v2 = self._vertex[(idx + 2) % 3] - self._vertex[idx]
        v2_len = np.linalg.norm(v2)
        cos_v2 = np.abs(np.dot(v1, v2) / v_len)
        sin_v2 = np.sqrt(v2_len * v2_len - cos_v2 * cos_v2)
        tex_coord[(idx + 2) % 3] = np.array(
            [sin_v2 / v_len, cos_v2 / v_len], dtype=np.float32
        )
        return tex_coord

    def _ray_intersect(self, ray: Ray) -> Tuple[np.float32, Intersection]:
        """
        alpha (v1 - v0) + beta (v2 - v0) + v0 = o + t.d
        (v1 - v0, v2 - v0, -d) @ (alpha, beta, t) = o - v0
        alpha, beta, t = inv((v1 - v0, v2 - v0, -d)) @ (o - v0)
        """
        ray_t = Ray.transform(ray, self._inv_transform)
        v0, v1, v2 = self._vertex
        cofficient = np.array([v1 - v0, v2 - v0, -ray_t.dir], dtype=np.float32)
        b = ray_t.pos - v0
        det = det3x3(cofficient)
        if np.isclose(det, 0):
            return INF, None

        inv_det = 1 / det

        alpha = det3x3(np.array([b, v2 - v0, -ray_t.dir], dtype=np.float32)) * inv_det
        if alpha < 0 or alpha > 1:
            return INF, None

        beta = det3x3(np.array([v1 - v0, b, -ray_t.dir], dtype=np.float32)) * inv_det
        if beta < 0 or beta > 1:
            return INF, None

        gamma = alpha + beta
        if gamma < 0 or gamma > 1:
            return INF, None

        t = det3x3(np.array([v1 - v0, v2 - v0, b], dtype=np.float32)) * inv_det
        if t < 0:
            return INF, None

        pos = ray.at(t)
        n_t = normalize(np.cross(v1 - v0, v2 - v0))
        uv = np.transpose(self._tex_coord) @ np.array([gamma, alpha, beta])
        return t, Intersection(pos, self.normal_to_world(n_t), uv, self.material)
