from abc import ABC, abstractmethod

from .bounding_box import AABB
from .ray import Ray


class RayIntersectObject(ABC):
    @abstractmethod
    def ray_intersect(self, ray: Ray):
        pass

    @abstractmethod
    def ray_intersect_cost(self):
        pass

    @property
    def bounding_box(self) -> AABB:
        return AABB()
