import numpy as np

from ..light import Light, LightSample
from ..typing import Vec3f, Vec2f


class PointLight(Light):
    def __init__(self, intensity: Vec3f, pos: Vec3f):
        super().__init__(Light.Type.DELTA_AREA)
        self.__intensity = intensity
        self.pos = pos

    def sample(self, target: Vec3f, u: Vec2f) -> LightSample:
        light_sample = LightSample()
        dist = target - self.pos
        dist_len = np.linalg.norm(dist)
        light_sample.wo = dist / dist_len
        light_sample.pdf = 1
        light_sample.le = self.__intensity / (dist_len * dist_len)
        return light_sample
