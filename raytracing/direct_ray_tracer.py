from enum import Enum
from typing import List

import numpy as np

from .constants import EPSILON, zero3f, ZERO3F
from .light import Light, LightSample
from .material import MaterialSample
from .ray import Ray
from .ray_intersect_object import RayIntersectObject
from .intersection import Intersection
from .transform import transform_dir
from .typing import Vec3f


class DirectRayTracer:
    class SampleStrategy(Enum):
        UNIFORM_SAMPLE_ALL = 1
        UNIFORM_SAMPLE_ONE = 2

    def __init__(self, objects: List[RayIntersectObject], lights: List[Light]):
        self.objects = objects
        self.lights = lights

    def render(self, view_ray: Ray, sample_strategy: SampleStrategy) -> Vec3f:
        inter = self.ray_intersect_objects(view_ray)
        if inter is None:
            return zero3f()
        if sample_strategy == DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ALL:
            lo = zero3f()
            for light in self.lights:
                lo += self.uniform_sample_light(inter, -view_ray.dir, light)
            return lo
        elif sample_strategy == DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ONE:
            light = self.lights[np.random.randint(len(self.lights))]
            return len(self.lights) * self.uniform_sample_light(
                inter, -view_ray.dir, light
            )
        return zero3f()

    def uniform_sample_light(self, intersection: Intersection, wo: Vec3f, light: Light):
        lo = zero3f()
        for _ in range(light.n_samples):
            lo += self.eva(intersection, wo, light) / light.n_samples
        return lo

    def eva(self, intersection: Intersection, wo: Vec3f, light: Light) -> Vec3f:
        local_wo = transform_dir(intersection.world_to_local, wo)
        lo = zero3f()
        # Sample light.
        light_sample = light.sample(intersection.pos, np.random.rand(2))
        if not np.allclose(light_sample.le, ZERO3F) and light_sample.pdf > 0:
            shadow_ray = Ray(
                intersection.pos + EPSILON * intersection.n, -light_sample.wo
            )
            block = self.ray_intersect_objects(shadow_ray)
            if block is None:
                local_wi = transform_dir(intersection.world_to_local, -light_sample.wo)
                scattering_pdf = intersection.pdf(local_wi, local_wo)
                scattering_f = intersection.f(local_wi, local_wo)
                if not np.allclose(scattering_f, ZERO3F) and scattering_pdf > 0:
                    lo += (
                        light_sample.le
                        * scattering_f
                        * np.abs(local_wi[2])
                        / light_sample.pdf
                    )
        return lo

    def ray_intersect_objects(self, ray: Ray) -> Intersection:
        inter = None
        for object in self.objects:
            inter_i = object.ray_intersect(ray)
            if not inter_i is None:
                inter = inter_i
        return inter
