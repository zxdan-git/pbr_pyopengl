from enum import Enum

import numpy as np

from ..constants import zero3f, ZERO3F
from ..importance_sampling_util import power_2_heuristic
from ..intersection import Intersection
from ..light import Light
from ..ray import Ray
from ..ray_tracer import RayTracer
from ..scene import Scene
from ..transform import transform_dir
from ..typing import Vec3f


class DirectRayTracer(RayTracer):
    class SampleStrategy(Enum):
        UNIFORM_SAMPLE_ALL = 1
        UNIFORM_SAMPLE_ONE = 2

    def __init__(self, scene: Scene, sample_strategy: SampleStrategy):
        super().__init__(scene)
        self.sample_strategy = sample_strategy

    def render(self, view_ray: Ray) -> Vec3f:
        inter = self.scene.ray_intersect_objects(view_ray)
        if inter is None:
            return self.scene.light_le(view_ray)
        if self.sample_strategy == DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ALL:
            return self.uniform_sample_all(inter, -view_ray.dir)
        elif self.sample_strategy == DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ONE:
            return self.uniform_sample_one(inter, -view_ray.dir)
        return zero3f()

    def uniform_sample_all(self, intersection: Intersection, wo: Vec3f):
        lo = zero3f()
        for light in self.scene.lights:
            light_lo = zero3f()
            for _ in range(light.n_samples):
                light_lo += self.eva(intersection, wo, light)
            lo += light_lo / light.n_samples
        return lo

    def uniform_sample_one(self, intersection: Intersection, wo: Vec3f):
        lo = zero3f()
        light = self.scene.lights[np.random.randint(len(self.scene.lights))]
        for _ in range(light.n_samples):
            lo += self.eva(intersection, wo, light)
        return len(self.scene.lights) * lo / light.n_samples

    def eva(self, intersection: Intersection, wo: Vec3f, light: Light) -> Vec3f:
        lo = zero3f()
        # Sample light.
        light_sample = light.sample(intersection.pos, np.random.rand(2))
        if not np.allclose(light_sample.le, ZERO3F) and not np.isclose(
            light_sample.pdf, 0
        ):
            material_sample = intersection.get_mat_sample(-light_sample.wo, wo)
            abs_cos_i = np.abs(material_sample.local_wi[2])
            if (
                not np.allclose(material_sample.f, ZERO3F)
                and not np.isclose(abs_cos_i, 0)
                and not np.isclose(material_sample.pdf, 0)
            ):
                wi_out = 1
                if material_sample.local_wi[2] < 0:
                    wi_out = -1
                shadow_ray = intersection.shoot_ray(-light_sample.wo, light_sample.t)
                block = self.scene.ray_intersect_objects(shadow_ray)
                if block is None:
                    weight = 1
                    if light.type == Light.Type.AREA:
                        weight = power_2_heuristic(
                            1, light_sample.pdf, 1, material_sample.pdf
                        )
                    lo += (
                        weight
                        * light_sample.le
                        * material_sample.f
                        * abs_cos_i
                        / light_sample.pdf
                    )

        # Sample material.
        if light.type == Light.Type.AREA:
            material_sample = intersection.sample_mat(wo, np.random.rand(2))
            abs_cos_i = np.abs(material_sample.local_wi[2])
            if (
                not np.allclose(material_sample.f, ZERO3F)
                and not np.isclose(material_sample.pdf, 0)
                and not np.isclose(abs_cos_i, 0)
            ):
                ray = intersection.shoot_ray(
                    material_sample.wi,
                )
                light_sample = light.get_sample(ray)
                if not np.allclose(light_sample.le, ZERO3F) and not np.isclose(
                    light_sample.pdf, 0
                ):
                    block = self.scene.ray_intersect_objects(ray)
                    if block is None:
                        weight = power_2_heuristic(
                            1, material_sample.pdf, 1, light_sample.pdf
                        )
                        lo += (
                            weight
                            * light_sample.le
                            * material_sample.f
                            * abs_cos_i
                            / material_sample.pdf
                        )

        return lo
