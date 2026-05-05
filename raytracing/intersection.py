from typing import List

import numpy as np

from .constants import zero3f, zero2f
from .material import Material, MaterialSample
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

    @property
    def world_to_local(self):
        return self.__world_to_local.copy()

    @property
    def local_to_world(self):
        return self.__local_to_world.copy()

    def sample_mat(self, wo: Vec3f, u: Vec2f) -> MaterialSample:
        if self.mat is None:
            return MaterialSample()
        local_wo = transform_dir(self.world_to_local, wo)
        material_sample = self.mat.sample(self.uv, local_wo, u)
        material_sample.wi = transform_dir(self.local_to_world, material_sample.wi)
        return material_sample

    def pdf(self, wi: Vec3f, wo: Vec3f):
        if self.mat is None:
            return 0
        return self.mat.pdf(self.uv, wi, wo)

    def f(self, wi: Vec3f, wo: Vec3f):
        if self.mat is None:
            return zero3f()
        return self.mat.f(self.uv, wi, wo)
