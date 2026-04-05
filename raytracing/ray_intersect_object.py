from abc import ABC, abstractmethod

import numpy as np

from .bounding_box import AABB
from .intersection import Intersection
from .ray import Ray


class RayIntersectObject(ABC):
    @abstractmethod
    def ray_intersect(self, ray: Ray) -> Intersection:
        pass

    @abstractmethod
    def ray_intersect_cost(self):
        pass

    @property
    def bounding_box(self) -> AABB:
        return AABB()
