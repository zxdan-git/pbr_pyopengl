import numpy as np
from pathlib import Path
from typing import Tuple

from ..bounding_box import AABB
from ..bounding_volume_hierarchy import BVH
from ..distributon import Distribution1D
from ..intersection import Intersection
from ..ray import Ray
from ..ray_intersect_object import RayIntersectObject
from ..shape import Shape
from ..shapes.triangle import Triangle
from ..typing import Vec2f, vec3f, vec2f, Array_u, Array_f, array_f, array_u


def get_line_index_from_face_index(face_index):
    if len(face_index) % 3 != 0:
        raise ValueError("face_index length must be a multiple of 3")

    edges = set()
    for i in range(0, len(face_index), 3):
        v0, v1, v2 = face_index[i : i + 3]
        edges.add(tuple(sorted((v0, v1))))
        edges.add(tuple(sorted((v1, v2))))
        edges.add(tuple(sorted((v2, v0))))

    line_index = []
    for v0, v1 in edges:
        line_index.extend((v0, v1))
    return line_index


class TriangleObject(RayIntersectObject):
    def __init__(self, mesh: Shape, face_indices: Array_u, tex_indices: Array_u):
        self.mesh = mesh
        if face_indices.shape != (3,) or tex_indices.shape != (3,):
            raise ValueError("3 index is needed for a triangle")
        self.v_idx_1, self.v_idx_2, self.v_idx_3 = face_indices
        self.t_idx_1, self.t_idx_2, self.t_idx_3 = tex_indices
        self._update_area()
        self._update_bounding_box()

    def on_mesh_transformed(self):
        self._update_area()
        self._update_bounding_box()

    def sample(self, u: Vec2f) -> Intersection:
        triangle = self._get_triangle()
        return triangle.sample(u)

    def ray_intersect(self, ray: Ray) -> Intersection:
        triangle = self._get_triangle()
        return triangle.ray_intersect(ray)

    def ray_intersect_cost(self):
        return 1

    @property
    def bounding_box(self) -> AABB:
        return self.bbx

    def _get_triangle(self) -> Triangle:
        vertex = self.mesh.transformed_vertex
        tex_coord = self.mesh.tex_coord
        triangle = Triangle(
            vertex[self.v_idx_1],
            vertex[self.v_idx_2],
            vertex[self.v_idx_3],
            tex_coord[self.t_idx_1],
            tex_coord[self.t_idx_2],
            tex_coord[self.t_idx_3],
        )
        triangle.material = self.mesh.material
        return triangle

    def _update_area(self):
        v1 = self.mesh.transformed_vertex[self.v_idx_1]
        v2 = self.mesh.transformed_vertex[self.v_idx_2]
        v3 = self.mesh.transformed_vertex[self.v_idx_3]
        self.area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    def _update_bounding_box(self):
        self.bbx = AABB()
        vertex = self.mesh.transformed_vertex
        for idx in [self.v_idx_1, self.v_idx_2, self.v_idx_3]:
            self.bbx.embrace(vertex[idx])


class TriangleMesh(Shape):
    def __init__(
        self,
        vertex: Array_f,
        tex_coord: Array_f,
        face_index: Array_u,
        tex_index: Array_u,
        bvh_type: BVH.Type = BVH.Type.SAH,
    ):
        super().__init__()
        self._vertex = vertex
        self._tex_coord = tex_coord
        self._face_index = face_index
        self._tex_index = tex_index
        if self._face_index.shape[0] != self._tex_index.shape[0]:
            raise ValueError("number of face index and textcoord index should be equal")
        if self._face_index.shape[0] % 3 != 0:
            raise ValueError("number of face index should be divided by 3")
        self._line_index = get_line_index_from_face_index(face_index)
        self._objects = [
            TriangleObject(self, face_indices, tex_indices)
            for face_indices, tex_indices in zip(
                np.reshape(self._face_index, (-1, 3)),
                np.reshape(self._tex_index, (-1, 3)),
            )
        ]
        # Always update the distribution after BVH since the BVH will recorder the objects.
        self._bvh_type = bvh_type
        self._bvh = BVH(
            self._bvh_type,
            self._objects,
        )
        self._area_distribution = Distribution1D([obj.area for obj in self._objects])

    @staticmethod
    def create_mesh_from_obj(obj_path: str, bvh_type: BVH.Type = BVH.Type.SAH):
        if Path(obj_path).suffix.lower() != ".obj":
            raise ValueError("Expect .obj file")
        vertex = []
        tex_coord = []
        face_index = []
        tex_index = []
        with open(obj_path, "r") as obj_file:
            lines = obj_file.readlines()
            for line in lines:
                data = line.rstrip().split(" ")
                if len(data) == 4 and data[0] == "v":
                    vertex.append(vec3f(float(data[1]), float(data[2]), float(data[3])))
                elif len(data) == 3 and data[0] == "vt":
                    tex_coord.append(vec2f(float(data[1]), float(data[2])))
                elif len(data) == 4 and data[0] == "f":
                    for index in data[1:]:
                        index_data = index.split("/")
                        if len(index_data) == 3:
                            face_index.append(int(index_data[0]) - 1)
<<<<<<< HEAD
                            tex_index.append(int(index_data[2]) - 1)
=======
                            if index_data[1] == "":
                                tex_index.append(0)
                            else:
                                tex_index.append(int(index_data[1]) - 1)
            if len(tex_coord) == 0:
                tex_coord.append(vec2f(0, 0))
>>>>>>> 3e0d691 (create triangle mesh)
            return TriangleMesh(
                array_f(vertex),
                array_f(tex_coord),
                array_u(face_index),
                array_u(tex_index),
                bvh_type,
            )

    def sample(self, u: Vec2f) -> Intersection:
        idx = self._area_distribution.sample(u[0])
        if idx < 0 or idx >= len(self._objects):
            raise ValueError("Wrong sampled triangle idx in triangle mesh.")
        cdfs = self._area_distribution.cdfs
        remap_u0 = (u[0] - cdfs[idx]) / (cdfs[idx + 1] - cdfs[idx])
        remap_u0 = np.clip(remap_u0, 0, 1)
        return self._objects[idx].sample(vec2f(remap_u0, u[1]))

    def _ray_intersect(self, ray: Ray) -> Tuple[np.float32, Intersection]:
        intersection = self._bvh.ray_intersect(ray)
        return ray.t_max, intersection

    def _on_transform_updated(self):
        for obj in self._objects:
            obj.on_mesh_transformed()

        self._bvh = BVH(
            self._bvh_type,
            self._objects,
        )
        self._area_distribution = Distribution1D([obj.area for obj in self._objects])
