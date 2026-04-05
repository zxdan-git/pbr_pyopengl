import numpy as np

from ..bxdf import BxDF
from ..constants import zero3f
from ..material import Material, MaterialSample
from ..typing import Vec2f, Vec3f


class MaterialOne(Material):
    def __init__(self, bxdf: BxDF):
        super().__init__()
        if not bxdf is None:
            self.bxdfs.append(bxdf)

    def sample(self, uv: Vec2f, wo: Vec3f, u: Vec2f) -> MaterialSample:
        if len(self.bxdfs) == 0:
            return None
        mat_sample = MaterialSample()
        bxdf_sample = self.bxdfs[0].sample(wo, u)
        mat_sample.wi = bxdf_sample.wi
        mat_sample.f = bxdf_sample.f
        mat_sample.pdf = bxdf_sample.pdf
        return mat_sample

    def pdf(self, uv: Vec2f, wi: Vec3f, wo: Vec3f) -> np.float32:
        if len(self.bxdfs) == 0:
            return 0
        return self.bxdfs[0].pdf(wi, wo)

    def f(self, uv: Vec2f, wi: Vec3f, wo: Vec3f) -> Vec3f:
        if len(self.bxdfs) == 0:
            return zero3f()
        return self.bxdfs[0].f(wi, wo)
