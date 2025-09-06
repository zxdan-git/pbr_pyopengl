import numpy as np

from .bounding_box import AABB
from .bvh_util.build_node import BuildNode
from .bvh_util.partition_util import (
    mid_point_partition_on_target_axis,
    equal_partition_on_target_axis,
    sah_partition_on_target_aixs,
)
from .bvh_util.recursive_build import build_by_axis_spread
from .bvh_util.hybrid_build import build_with_morton_code
from .interval import Interval
from .ray import Ray
from .ray_intersect_object import RayIntersectObject
from typing import List, Callable
from enum import Enum


class BVH(RayIntersectObject):
    class Type(Enum):
        MID_POINT = 1
        EQUAL_COUNT = 2
        SAH = 3
        MORTON_CODE = 4

    def __init__(self, type, objects):
        self.__type: BVH.Type = type
        self.__objects: List[RayIntersectObject] = objects
        self.__root: BuildNode = None
        if type == BVH.Type.MORTON_CODE:
            self.__root = build_with_morton_code(self.__objects)
        else:
            self.__root = self._recursive_build()

    @property
    def root(self):
        return self.__root

    @property
    def type(self):
        return self.__type

    @property
    def bounding_box(self) -> AABB:
        if self.__root is None:
            return AABB()
        return self.__root.bounding_box

    def ray_intersect(self, ray: Ray):
        return self.__root.ray_intersect(ray)

    def ray_intersect_cost(self):
        if self.__root is None:
            return 0
        return self.__root.ray_intersect_cost()

    def _recursive_build(self):
        partition_func: Callable[
            [List[RayIntersectObject], int, int, int, Interval], int
        ] = None
        if self.type == BVH.Type.MID_POINT:
            partition_func = mid_point_partition_on_target_axis
        elif self.type == BVH.Type.EQUAL_COUNT:
            partition_func = equal_partition_on_target_axis
        elif self.type == BVH.Type.SAH:
            partition_func = sah_partition_on_target_aixs
        else:
            raise ValueError("Invalid BVH type for recursive building.")

        return build_by_axis_spread(
            self.__objects, 0, len(self.__objects), partition_func
        )
