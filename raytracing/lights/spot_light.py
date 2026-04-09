import numpy as np

from ..light import Light, LightSample
from ..shape_sample_util import (
    uniform_sample_direction_in_cone,
    uniform_sample_cone_pdf,
)
from ..transform import world_to_local_from_single_dir
from ..typing import Vec3f, Vec2f, Mat4f
from ..util import normalize


class SpotLight(Light):
    def __init__(self, intensity: Vec3f, pos: Vec3f, dir: Vec3f, theta_max: np.float32):
        super().__init__()
        self.__intensity = intensity
        self.__pos = pos
        self.__dir = normalize(dir)
        self.__theta_max = theta_max
        self.__world_to_local: Mat4f = world_to_local_from_single_dir(self.__dir)

    def sample(self, target: Vec3f, u: Vec2f) -> LightSample:
        light_sample = LightSample()
        dist = target - self.__pos
        dist_len = np.linalg.norm(dist)
        if dist_len == 0:
            return light_sample
        dist_dir = dist / dist_len
        if np.dot(dist_dir, self.__dir) < np.cos(self.__theta_max):
            return light_sample
        light_sample.le = self.__intensity / dist_len / dist_len
        light_sample.pdf = uniform_sample_cone_pdf(self.__theta_max)
        light_sample.wo = dist_dir
        return light_sample

    def pdf(self, wo: Vec3f) -> np.float32:
        return 0
