import numpy as np
from typing import List

from ..bounding_box import AABB
from ..intersection import Intersection
from ..ray import Ray
from ..ray_intersect_object import RayIntersectObject


class BuildNode(RayIntersectObject):
    def __init__(self):
        self.left: BuildNode = None
        self.right: BuildNode = None
        self.bbx = AABB()
        self.objects: List[RayIntersectObject] = []

    def ray_intersect(self, ray: Ray) -> Intersection:
        if not self.bbx.ray_intersect(ray):
            return None

        # If it is a leaf node, directly check the intersections from its
        # objects.
        intersection = None
        if len(self.objects) != 0:
            for object in self.objects:
                ray_intersection = object.ray_intersect(ray)
                if not ray_intersection is None:
                    intersection = ray_intersection
            return intersection

        # Otherwise, continue tranversal.
        for child in [self.left, self.right]:
            if not child is None:
                ray_intersection = child.ray_intersect(ray)
                if not ray_intersection is None:
                    intersection = ray_intersection
        return intersection

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
