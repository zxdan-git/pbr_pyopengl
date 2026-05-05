from abc import abstractmethod
from enum import IntFlag
from typing import Tuple

import numpy as np

from .bounding_box import AABB
from .constants import INF
from .intersection import Intersection
from .material import Material
from .ray import Ray
from .ray_intersect_object import RayIntersectObject
from .transform import transform_dir, transform_pos
from .util import normalize
from .typing import Vec3f, Vec2f, array_f, array_u


class Shape(RayIntersectObject):
    class PaintMode(IntFlag):
        FACE = 1
        LINE = 2
        FACE_AND_LINE = FACE | LINE

    def __init__(self):
        self._vertex = np.empty((0, 3))
        self.__transformed_vertex = np.empty((0, 3))
        self._face_index = array_u([])
        self._line_index = array_u([])
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
        self.__transformed_vertex = array_f(
            [transform_pos(self.transform, v) for v in self._vertex]
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
