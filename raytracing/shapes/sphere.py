import numpy as np
from typing import Tuple

from ..bounding_box import AABB
from ..constants import INF
from ..shape import Shape
from ..intersection import Intersection
from ..ray import Ray
from ..typing import Array_u, Vec2f, Vec3f, vec2f, vec3f, array_f, array_u
from ..util import normalize

from .shape_sample_util import uniform_sample_sphere, uniform_sphere_pdf


class Sphere(Shape):
    def __init__(self, nu, nv):
        super().__init__()
        self._generate_vertex(nu, nv)
        self._generate_tex_coord(nu, nv)
        self._face_index = self._generate_face_index(nu, nv)
        self._tex_index = self._generate_face_index(nu, nv)
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
        return vec2f(0.5 + 0.5 * phi / np.pi, theta / np.pi)

    def _generate_vertex(self, nu, nv):
        vertex = []
        for i in range(nu + 1):
            theta = np.pi * float(i) / nu
            y = np.cos(theta)
            r = np.sin(theta)
            for j in range(nv + 1):
                phi = 2 * np.pi * float(j) / nv
                vertex.append(vec3f(r * np.sin(phi), y, r * np.cos(phi)))
        self._vertex = array_f(vertex)

    def _generate_tex_coord(self, nu, nv):
        tex_coord = []
        for i in range(nu + 1):
            for j in range(nv + 1):
                tex_coord.append(vec2f(float(i) / nu, float(j) / nv))
        self._tex_coord = array_f(tex_coord)

    def _generate_face_index(self, nu, nv) -> Array_u:
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
        return array_u(index)

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
        self._line_index = array_u(index)
