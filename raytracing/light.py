from abc import ABC, abstractmethod
import numpy as np

from .constants import zero3f
from .typing import Vec2f, Vec3f


class LightSample:
    def __init__(self):
        self.wo = np.array([0, 0, 1], dtype=np.float32)
        self.pdf = 0
        self.le = zero3f()


class Light(ABC):
    def __init__(self):
        self.n_samples = 1

    @abstractmethod
    def sample(self, target: Vec3f, u: Vec2f) -> LightSample:
        return LightSample()

    @abstractmethod
    def pdf(self, wo: Vec3f) -> np.float32:
        return 0
