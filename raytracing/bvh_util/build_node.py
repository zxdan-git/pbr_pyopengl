import numpy as np
from typing import List

from ..bounding_box import AABB
from ..ray import Ray
from ..ray_intersect_object import RayIntersectObject


class BuildNode(RayIntersectObject):
    def __init__(self):
        self.left: BuildNode = None
        self.right: BuildNode = None
        self.bbx = AABB()
        self.objects: List[RayIntersectObject] = []

    def ray_intersect(self, ray: Ray):
        if not self.bbx.ray_intersect(ray):
            return None

        if len(self.objects):
            intersect = False
            for object in self.objects:
                if not object.ray_intersect(ray) is None:
                    intersect = True

            # Note: we should return ray.t_max to get the nearest
            # intersection.
            if intersect:
                return ray.t_max
            else:
                return None

        for child in [self.left, self.right]:
            if not child is None:
                intersect = child.ray_intersect(ray)
                if not intersect is None:
                    return intersect
        return None

    def ray_intersect_cost(self):
        # Using SAH as the intersection cost.
        if len(self.objects) == 0:
            cost = 0.125
            inv_total_area = 1 / self.bbx.surface_area()
            for child in [self.left, self.right]:
                if not child is None:
                    cost += (
                        child.bbx.surface_area()
                        * inv_total_area
                        * child.ray_intersect_cost()
                    )
            return cost
        return np.sum([object.ray_intersect_cost() for object in self.objects])

    @property
    def bounding_box(self):
        return self.bbx
