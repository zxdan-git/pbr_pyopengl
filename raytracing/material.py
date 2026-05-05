from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .bxdf import BxDF
from .constants import zero3f
from .typing import Vec2f, Vec3f, vec3f


class MaterialSample:
    def __init__(self):
        self.wi = vec3f(0, 0, 1)
        self.pdf = 0
        self.f = zero3f()


class Material(ABC):
    def __init__(self):
        self.bxdfs: List[BxDF] = []

    @abstractmethod
    def sample(self, uv: Vec2f, wo: Vec3f, u: Vec2f) -> MaterialSample:
        return MaterialSample()

    @abstractmethod
    def pdf(self, uv: Vec2f, wi: Vec3f, wo: Vec3f) -> np.float32:
        return 0

    @abstractmethod
    def f(self, uv: Vec2f, wi: Vec3f, wo: Vec3f) -> Vec3f:
        return zero3f()
