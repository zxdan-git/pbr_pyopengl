from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from .constants import zero3f
from .ray import Ray
from .typing import Vec2f, Vec3f, vec3f


class LightSample:
    def __init__(self):
        self.wo = vec3f(0, 0, 1)
        self.pdf = 0
        self.le = zero3f()


class Light(ABC):
    class Type(Enum):
        DELTA_AREA = 1
        DELTA_DIR = 2
        AREA = 3

    def __init__(self, type: Type):
        self.n_samples = 1
        self.type = type

    @abstractmethod
    def sample(self, target: Vec3f, u: Vec2f) -> LightSample:
        return LightSample()

    def pdf(self, target: Vec3f, wi: Vec3f) -> np.float32:
        return 0

    def le(self, ray: Ray) -> Vec3f:
        return zero3f()

    def is_delta(self) -> bool:
        return self.type == Light.Type.DELTA_AREA or self.type == Light.Type.DELTA_DIR
