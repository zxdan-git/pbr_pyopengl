from typing import List

import numpy as np

from .constants import EPSILON, INF, zero3f, zero2f
from .material import Material, MaterialSample
from .ray import Ray
from .transform import world_to_local_from_single_dir, transform_dir
from .typing import Vec3f, Vec2f, vec3f


class Intersection:
    def __init__(
        self,
        pos: Vec3f = zero3f,
        n: Vec3f = vec3f(0, 1, 0),
        uv: Vec2f = zero2f(),
        mat: Material = None,
        sample_pdf=-1,
    ):
        self.pos = pos
        self.__n = n
        self.uv = uv
        self.mat = mat
        self.sample_pdf = sample_pdf
        self.__world_to_local = world_to_local_from_single_dir(self.n)
        self.__local_to_world = np.transpose(self.__world_to_local)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, new_n):
        self.__n = new_n
        self.__world_to_local = world_to_local_from_single_dir(self.n)
        self.__local_to_world = np.transpose(self.__world_to_local)

    def sample_mat(self, wo: Vec3f, u: Vec2f) -> MaterialSample:
        if self.mat is None:
            return MaterialSample()
        local_wo = self._world_to_local_dir(wo)
        material_sample = self.mat.sample(self.uv, local_wo, u)
        material_sample.wi = self._local_to_world_dir(material_sample.local_wi)
        return material_sample

    def get_mat_sample(self, wi: Vec3f, wo: Vec3f) -> MaterialSample:
        mat_sample = MaterialSample()
        mat_sample.wi = wi
        mat_sample.local_wi = self._world_to_local_dir(wi)
        local_wo = self._world_to_local_dir(wo)
        mat_sample.f = self._f(mat_sample.local_wi, local_wo)
        mat_sample.pdf = self._pdf(mat_sample.local_wi, local_wo)
        return mat_sample

    def shoot_ray(self, dir, t=INF) -> Ray:
        out = 1
        if np.dot(dir, self.n) < 0:
            out = -1
        return Ray(self.pos + out * EPSILON * self.n, dir, t)

    def _pdf(self, local_wi: Vec3f, local_wo: Vec3f):
        if self.mat is None:
            return 0
        return self.mat.pdf(self.uv, local_wi, local_wo)

    def _f(self, local_wi: Vec3f, local_wo: Vec3f):
        if self.mat is None:
            return zero3f()
        return self.mat.f(self.uv, local_wi, local_wo)

    def _world_to_local_dir(self, dir):
        return transform_dir(self.__world_to_local, dir)

    def _local_to_world_dir(self, dir):
        return transform_dir(self.__local_to_world, dir)
