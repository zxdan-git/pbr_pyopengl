from enum import Enum
from typing import List

import numpy as np

from .constants import EPSILON, zero3f, ZERO3F
from .importance_sampling_util import power_2_heuristic
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
            return self.light_le(view_ray)
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
        if not np.allclose(light_sample.le, ZERO3F) and not np.isclose(
            light_sample.pdf, 0
        ):
            shadow_ray = Ray(
                intersection.pos + EPSILON * intersection.n, -light_sample.wo
            )
            local_wi = transform_dir(intersection.world_to_local, -light_sample.wo)
            cos_i = np.abs(local_wi[2])
            scattering_pdf = intersection.pdf(local_wi, local_wo)
            scattering_f = intersection.f(local_wi, local_wo)
            if (
                not np.allclose(scattering_f, ZERO3F)
                and not np.isclose(cos_i, 0)
                and not np.isclose(scattering_pdf, 0)
            ):
                block = self.ray_intersect_objects(shadow_ray)
                if block is None:
                    weight = 1
                    if light.type == Light.Type.AREA:
                        weight = power_2_heuristic(
                            1, light_sample.pdf, 1, scattering_pdf
                        )
                    lo += (
                        weight
                        * light_sample.le
                        * scattering_f
                        * cos_i
                        / light_sample.pdf
                    )

        # Sample material.
        if light.type == Light.Type.AREA:
            material_sample = intersection.sample_mat(wo, np.random.rand(2))
            cos_i = np.abs(np.dot(material_sample.wi, intersection.n))
            if (
                not np.allclose(material_sample.f, ZERO3F)
                and not np.isclose(material_sample.pdf, 0)
                and not np.isclose(cos_i, 0)
            ):
                ray = Ray(intersection.pos, material_sample.wi)
                light_le = light.le(ray)
                light_pdf = light.pdf(intersection.pos, material_sample.wi)
                if not np.allclose(light_le, ZERO3F) and not np.isclose(light_pdf, 0):
                    block = self.ray_intersect_objects(ray)
                    if block is None:
                        weight = power_2_heuristic(1, material_sample.pdf, 1, light_pdf)
                        lo += (
                            weight
                            * light_le
                            * material_sample.f
                            * cos_i
                            / material_sample.pdf
                        )

        return lo

    def ray_intersect_objects(self, ray: Ray) -> Intersection:
        inter = None
        for object in self.objects:
            inter_i = object.ray_intersect(ray)
            if not inter_i is None:
                inter = inter_i
        return inter

    def light_le(self, view_ray: Ray):
        le = zero3f()
        for light in self.lights:
            if light.is_delta():
                continue
            le += light.le(view_ray)
        return le
