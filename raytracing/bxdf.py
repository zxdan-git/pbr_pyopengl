from abc import ABC, abstractmethod
import numpy as np

from .constants import zero3f
from .typing import Vec2f, Vec3f


class BxDFSample:
    def __init__(self):
        self.wi = np.array([0, 0, 1], dtype=np.float32)
        self.pdf = 0
        self.f = zero3f()


class BxDF(ABC):
    @abstractmethod
    def sample(self, wo: Vec3f, u: Vec2f) -> BxDFSample:
        return BxDFSample()

    @abstractmethod
    def pdf(self, wi: Vec3f, wo: Vec3f) -> np.float32:
        return 0

    @abstractmethod
    def f(self, wi: Vec3f, wo: Vec3f) -> Vec3f:
        return zero3f()
