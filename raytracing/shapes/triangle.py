import numpy as np
from typing import Tuple

from ..bounding_box import AABB
from ..constants import INF
from ..shape import Shape
from ..intersection import Intersection
from ..ray import Ray
from ..typing import Vec2f, Vec3f
from ..util import normalize, det3x3

from .shape_sample_util import uniform_sample_triangle


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
