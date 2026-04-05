import numpy as np

from ..bxdf import BxDF, BxDFSample
from ..constants import one3f
from ..shape_sample_util import cosine_sample_hemisphere, cosine_sample_hemisphere_pdf
from ..typing import Vec2f, Vec3f


class Lambertain(BxDF):
    def sample(self, wo: Vec3f, u: Vec2f) -> BxDFSample:
        bxdf_sample = BxDFSample()
        bxdf_sample.wi = cosine_sample_hemisphere(u)
        bxdf_sample.pdf = cosine_sample_hemisphere_pdf(bxdf_sample.wi)
        bxdf_sample.f = one3f() / np.pi
        return bxdf_sample

    def pdf(self, wi: Vec3f, wo: Vec3f) -> np.float32:
        if wi[2] * wo[2] < 0:
            return 0
        return cosine_sample_hemisphere_pdf(wi)

    def f(self, wi: Vec3f, wo: Vec3f) -> Vec3f:
        return one3f() / np.pi
