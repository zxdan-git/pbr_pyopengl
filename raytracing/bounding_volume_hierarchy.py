from typing import List, Callable
from enum import Enum

from .bounding_box import AABB
from .bvh_util.build_node import BuildNode
from .bvh_util.partition_util import (
    mid_point_partition_on_target_axis,
    equal_partition_on_target_axis,
    sah_partition_on_target_aixs,
)
from .bvh_util.recursive_build import build_by_axis_spread
from .bvh_util.hybrid_build import build_with_morton_code
from .intersection import Intersection
from .interval import Interval
from .ray import Ray
from .ray_intersect_object import RayIntersectObject


class BVH(RayIntersectObject):
    class Type(Enum):
        MID_POINT = 1
        EQUAL_COUNT = 2
        SAH = 3
        MORTON_CODE = 4

    def __init__(self, type, objects):
        self._type: BVH.Type = type
        self._objects: List[RayIntersectObject] = objects
        self._root: BuildNode = None
        if type == BVH.Type.MORTON_CODE:
            self._root = build_with_morton_code(self._objects)
        else:
            self._root = self._recursive_build()

    def copy_to(self, objects):
        if len(objects) != len(self._objects):
            raise ValueError("bvh copy target should have the same size of objects")
        bvh = self.__class__.__new__(self.__class__)
        bvh._objects = objects
        bvh._type = self._type
        if self._root is None:
            return bvh
        obj_maps = {a: b for a, b in zip(self._objects, objects)}
        bvh._root = BuildNode()
        stack = [(self._root, bvh._root)]
        while len(stack) != 0:
            ref, target = stack.pop()
            target.bbx = ref.bbx
            if not ref.left is None:
                target.left = BuildNode()
                stack.append((ref.left, target.left))
            if not ref.right is None:
                target.right = BuildNode()
                stack.append((ref.right, target.right))
            if len(ref.objects) > 0:
                target.objects = [obj_maps[obj] for obj in ref.objects]
        return bvh

    @property
    def root(self):
        return self._root

    @property
    def type(self):
        return self._type

    @property
    def bounding_box(self) -> AABB:
        if self._root is None:
            return AABB()
        return self._root.bounding_box

    def ray_intersect(self, ray: Ray) -> Intersection:
        return self._root.ray_intersect(ray)

    def ray_intersect_cost(self):
        if self._root is None:
            return 0
        return self._root.ray_intersect_cost()

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
            self._objects, 0, len(self._objects), partition_func
        )
