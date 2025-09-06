import numpy as np
from typing import List

from ..bounding_box import AABB
from ..ray_intersect_object import RayIntersectObject


class MortonObject:
    def __init__(self, object, morton):
        self.object = object
        self.morton = morton


def vec3_to_morton(vec3) -> np.uint32:
    def shift_3(v: np.uint32) -> np.uint32:
        v = (v | v << 16) & 0b00000011000000000000000011111111
        v = (v | v << 8) & 0b00000011000000001111000000001111
        v = (v | v << 4) & 0b00000011000011000011000011000011
        v = (v | v << 2) & 0b00001001001001001001001001001001
        return v

    morton = np.uint32(0)
    for i in range(3):
        morton += shift_3(vec3[i]) << i
    return morton


def generate_morton_objects_from_intersect_objects(
    objects: List[RayIntersectObject],
) -> List[MortonObject]:
    center_bbx = AABB()
    for object in objects:
        center_bbx.embrace(object.bounding_box.center())

    bits_per_dim = 10
    range_per_dim = 1 << bits_per_dim
    center_offsets = [
        np.int32(center_bbx.offset(object.bounding_box.center()) * range_per_dim)
        for object in objects
    ]

    return [
        MortonObject(object, vec3_to_morton(center_offset))
        for object, center_offset in zip(objects, center_offsets)
    ]


def partition_on_sorted_morton(objects: List[MortonObject], start, end, partition_bit):
    if start >= end or partition_bit < 0:
        return -1

    mask = 1 << partition_bit
    if (objects[start].morton & mask) == (objects[end - 1].morton & mask):
        return end

    search_start, search_end = start, end
    while search_start < search_end:
        mid = (search_start + search_end) // 2
        if (objects[search_start].morton & mask) == (objects[mid].morton & mask):
            search_start = mid + 1
        else:
            search_end = mid
    return search_start
