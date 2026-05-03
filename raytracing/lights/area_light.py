import numpy as np

from ..constants import zero3f
from ..light import Light, LightSample
from ..ray import Ray
from ..shape import Shape
from ..typing import Vec3f, Vec2f


class AreaLight(Light):
    def __init__(self, intensity: Vec3f, shape: Shape):
        super().__init__(Light.Type.AREA)
        self.__intensity = intensity
        self.__shape = shape

    @property
    def transform(self):
        return self.__shape.transform

    @transform.setter
    def transform(self, new_transform):
        self.__shape.transform = new_transform

    def sample(self, target: Vec3f, u: Vec2f) -> LightSample:
        light_sample = LightSample()
        intersection = self.__shape.sample_for_target(target, np.random.rand(2))
        if intersection is None:
            return LightSample()
        dist = target - intersection.pos
        dist_len = np.linalg.norm(dist)
        light_sample.wo = dist / dist_len
        light_sample.le = self.__intensity
        if np.dot(light_sample.wo, intersection.n) <= 0:
            light_sample.le = 0
        light_sample.pdf = self.pdf(target, -light_sample.wo)
        return light_sample

    def pdf(self, target: Vec3f, wi: Vec3f) -> np.float32:
        return self.__shape.sample_pdf_for_target(target, wi)

    def le(self, ray: Ray) -> Vec3f:
        intersection = self.__shape.ray_intersect(ray)
        if intersection is None:
            return zero3f()
        return self.__intensity
