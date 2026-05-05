import numpy as np

from ..bxdf import BxDF, BxDFSample
from ..typing import Vec2f, Vec3f, vec3f
from ..constants import INF, zero3f, one3f


class Specular(BxDF):
    def sample(self, wo: Vec3f, u: Vec2f) -> BxDFSample:
        """
        Lo = \int f(wi, wo) Li(wi) |cos(theta_i)| dwi

        According to the property of perfect specular, the only possible
        incident direction is the reflected direction. We could use delta
        function to make integral concentrate on the reflected direction:
        f(wi, wo) = g(wi, wo) \delta(wi - wr) where wr = reflected(wo)

        Since it is a perfect specular BxDF, no energy loses except the part
        aborbed by the BxDF.

        Lo = \int f(wi, wo) Li(wi) |cos(theta_i)| dwi
           = \int g(wi, wo) \delta(wi - wr) Li(wi) |cos(theta_i)| dwi
           = g(wr, wo) |cos(theta_r)| Li(wr) = L(wr)

        g(wr, wo) = 1 / |cos(theta_r)|
        f(wr, wo) = \delta(0) / |cos(theta_r)|

        Note, \delta(0) is infinity but will be cancelled during Monte Carlo
        integration, so f(wr, wo) = 1 / |cos(theta_r)| here.
        """
        mat_sample = BxDFSample()
        mat_sample.wi = vec3f(-wo[0], -wo[1], wo[2])
        mat_sample.pdf = INF  # delta(0)
        mat_sample.f = one3f() / mat_sample.wi[2]

    def pdf(self, wi: Vec3f, wo: Vec3f) -> np.float32:
        return 0

    def f(self, wi: Vec3f, wo: Vec3f) -> Vec3f:
        return zero3f()
