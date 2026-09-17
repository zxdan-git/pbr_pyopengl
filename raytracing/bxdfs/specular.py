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
        f(w_i, w_o) = g(w_i, w_o) \\delta(w_i - w_r) where w_r = reflected(w_o).

        Since it is a perfect specular BxDF, no energy loses except the part
        aborbed by the BxDF.

        dPhi_i = \\int\\int Li |cos(theta_i)| dw_i dA
        dPhi_o
            = \\int\\int\\int f(w_i, w_o) |cos(theta_i)cos(theta_o)| dw_i dw_o
                dA
            = \\int\\int\\int g(w_i, w_o) \\delta(wi - wr) |cos(theta_i)
                cos(theta_o)| dw_i dw_o dA
            = \\int\\int g(w_i, w_r) |cos(theta_i)cos(theta_r)| dw_i dw_o dA

        dPhi_o = Fr * dPhi_i, where Fr is the Fresnel factor which will be estimated by the Material.
        g(w_i, w_r) = Fr  / |cos(theta_r)|

        Note, \\delta(0) is infinity but will be cancelled during Monte Carlo
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
