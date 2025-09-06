import numpy as np
from typing import List, Callable

from ..bounding_box import AABB
from .build_node import BuildNode
from ..interval import Interval
from .morton_code_util import MortonObject, partition_on_sorted_morton
from ..ray_intersect_object import RayIntersectObject


def build_by_axis_spread(
    objects: List[RayIntersectObject],
    start,
    end,
    partition_func_on_axis: Callable[
        [List[RayIntersectObject], int, int, int, Interval], int
    ],
) -> BuildNode:
    node = BuildNode()
    if end <= start:
        return None

    # Build bounding box for the node.
    for object in objects[start:end]:
        node.bbx = AABB.union(node.bbx, object.bounding_box)

    # If there is only one object, directly create a leaf node.
    if end - start == 1:
        node.objects = [objects[start]]
        return node

    # Find the axis along which the object centers are most widely
    # distributed to perform partition.
    center_bbx = AABB()
    for i in range(start, end):
        center_bbx.embrace(objects[i].bounding_box.center())

    target_axis = center_bbx.get_max_axis()
    target_range = center_bbx.get_range(target_axis)

    # If all objects have the same center, stop partition and save them
    # in one leaf node. Otherwise the parition will not end.
    if np.isclose(target_range.lower, target_range.upper):
        node.objects = objects[start:end]
        return node

    # Do partition according to the type BVH type.
    mid = partition_func_on_axis(objects, start, end, target_axis, target_range)

    # If mid is not in between start and end, directly generate a leaf node.
    if mid == start or mid == end:
        node.objects = objects[start:end]
        return node

    # Recursively create child nodes.
    node.left = build_by_axis_spread(objects, start, mid, partition_func_on_axis)
    node.right = build_by_axis_spread(objects, mid, end, partition_func_on_axis)
    return node


def build_by_sorted_morton(
    morton_objects: List[MortonObject], start, end, partition_bit
):
    if start >= end:
        return None

    node = BuildNode()
    for morton_object in morton_objects[start:end]:
        node.bbx = AABB.union(node.bbx, morton_object.object.bounding_box)

    if partition_bit == -1 or end - start == 1:
        for morton_object in morton_objects[start:end]:
            node.objects.append(morton_object.object)
        return node

    mid = partition_on_sorted_morton(morton_objects, start, end, partition_bit)

    if mid == end:
        return build_by_sorted_morton(morton_objects, start, end, partition_bit - 1)

    node.left = build_by_sorted_morton(morton_objects, start, mid, partition_bit - 1)
    node.right = build_by_sorted_morton(morton_objects, mid, end, partition_bit - 1)
    return node
