from typing import List

import numpy as np

from .material import Material, MaterialSample
from .transform import world_to_local_from_single_dir
from .typing import Vec3f, Vec2f


class Intersection:
    def __init__(self, pos: Vec3f, n: Vec3f, uv: Vec2f, mat: Material):
        self.pos = pos
        self.__n = n
        self.uv = uv
        self.mat = mat
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
        return self.mat.sample(self.uv, wo, u)

    def pdf(self, wi: Vec3f, wo: Vec3f):
        return self.mat.pdf(self.uv, wi, wo)

    def f(self, wi: Vec3f, wo: Vec3f):
        return self.mat.f(self.uv, wi, wo)
