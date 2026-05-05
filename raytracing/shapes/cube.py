import numpy as np
from typing import Tuple

from ..bounding_box import AABB
from ..constants import INF, zero2f, zero3f, ZERO3F
from ..shape import Shape
from ..intersection import Intersection
from ..ray import Ray
from ..typing import Vec2f, vec2f, array_f, array_u


class Cube(Shape):
    def __init__(self):
        super().__init__()
        self._vertex = array_f(
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
            ]
        )

        self._face_index = array_u(
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
            ]
        )

        self._line_index = array_u(
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
            ]
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
        intersection.uv = vec2f(remapped_u0, u[1])
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
