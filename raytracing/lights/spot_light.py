import numpy as np

from ..light import Light, LightSample
from ..shape_sample_util import (
    uniform_sample_cone_pdf,
)
from ..typing import Vec3f, Vec2f
from ..util import normalize


class SpotLight(Light):
    def __init__(
        self,
        intensity: Vec3f,
        pos: Vec3f,
        dir: Vec3f,
        theta_max: np.float32,
        theta_decay: np.float32 = 0,
    ):
        super().__init__(Light.Type.DELTA_DIR)
        self.__intensity = intensity
        self.__pos = pos
        self.__dir = normalize(dir)
        self.__theta_max = theta_max
        self.__cos_theta_max = np.cos(theta_max)
        self.__cos_theta_decay = np.cos(theta_decay)

    def sample(self, target: Vec3f, u: Vec2f) -> LightSample:
        light_sample = LightSample()
        dist = target - self.__pos
        dist_len = np.linalg.norm(dist)
        if dist_len == 0:
            return light_sample
        dist_dir = dist / dist_len
        cos_theta = np.dot(dist_dir, self.__dir)
        if cos_theta < self.__cos_theta_max:
            return light_sample
        weight = 1
        if cos_theta < self.__cos_theta_decay:
            weight = np.power(
                (self.__cos_theta_decay - cos_theta)
                / (self.__cos_theta_decay - self.__cos_theta_max),
                4,
            )
        light_sample.le = weight * self.__intensity / dist_len / dist_len
        light_sample.pdf = uniform_sample_cone_pdf(self.__theta_max)
        light_sample.wo = dist_dir
        return light_sample
