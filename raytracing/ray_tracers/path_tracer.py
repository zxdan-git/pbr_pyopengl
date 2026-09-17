import numpy as np

from .direct_ray_tracer import DirectRayTracer

from ..constants import ZERO3F, zero3f, one3f
from ..intersection import Intersection
from ..ray import Ray
from ..ray_tracer import RayTracer
from ..scene import Scene
from ..typing import Vec3f
from ..util import rgb_to_luminance


class PathTracer(RayTracer):
    def __init__(self, scene: Scene, max_path_len):
        super().__init__(scene)
        self.max_path_len = max_path_len
        self.direct_tracer = DirectRayTracer(
            scene, DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ONE
        )

    def render(self, view_ray: Ray) -> Vec3f:
        path_len = 0
        throughput = one3f()

        # Handle the case when path length = 1
        path_len += 1
        lo = self.scene.light_le(view_ray)

        # Find a intersection of the object with the view ray.
        intersection = self.scene.ray_intersect_objects(view_ray)
        if intersection is None:
            return lo
        path_len += 1
        lo += self.eva(intersection, -view_ray.dir, path_len, throughput)
        return lo

    def eva(
        self, intersection: Intersection, wo: Vec3f, path_len: int, throughput: Vec3f
    ) -> Vec3f:
        if path_len >= self.max_path_len:
            return zero3f()

        mat_sample = intersection.sample_mat(wo, np.random.rand(2))
        abs_cos_i = np.abs(mat_sample.local_wi[2])
        if (
            np.allclose(mat_sample.f, ZERO3F)
            or np.isclose(mat_sample.pdf, 0)
            or np.isclose(abs_cos_i, 0)
        ):
            return zero3f()

        lo = zero3f()
        # Estimate the radiance of path with current length.
        lo += throughput * self.direct_tracer.uniform_sample_one(intersection, wo)

        next_inter = self.scene.ray_intersect_objects(
            intersection.shoot_ray(mat_sample.wi)
        )
        if next_inter is None:
            return lo

        path_len += 1
        throughput *= mat_sample.f * abs_cos_i / mat_sample.pdf
        # Use Russian roulette to decide if we should continue tracing.
        end_prob = max(0.05, 1.0 - rgb_to_luminance(throughput))
        if np.random.rand() < end_prob:
            return lo
        throughput /= 1 - end_prob
        lo += self.eva(next_inter, -mat_sample.wi, path_len, throughput)
        return lo
