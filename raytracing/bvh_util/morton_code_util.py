import numpy as np
from typing import List

from ..bounding_box import AABB
from ..ray_intersect_object import RayIntersectObject


class MortonObject:
    """
    A container that associates a geometric object with its Morton code.

    The Morton code is a technique for mapping multidimensional data to a
    one-dimensional value by interleaving the bits of the coordinates.

    For example, given a 3D point with coordinates x=5 (101), y=6 (110),
    and z=7 (111), the corresponding Morton code is created by interleaving
    their bits: 111 011 101.

    Sorting objects based on their Morton codes clusters them spatially,
    which is particularly useful for efficiently building a Bounding Volume
    Hierarchy (BVH).

    Note that we assume the coordinates values are no larger than 2^10.
    """

    def __init__(self, object, morton):
        self.object = object
        self.morton = morton


def vec3_to_morton(vec3) -> np.uint32:
    def shift_3(v: np.uint32) -> np.uint32:
        """
        A helper function to expands a 10-bit integer into a 30-bit integer by
        inserting two zeros between each bit. This prepares the integer for
        interleaving to create a Morton code.

        Since the value is not larger than 2^10, there will be at most 10 bits
        to move. For example the 10 bit:

        b9 b8 b7 b6 b5 b4 b3 b2 b1 b0

        will be spreaded out to a 30-bit integer:

        00 b9 00 b8 00 b7 00 b6 00 b5 00 b4 00 b3 00 b2 00 b1 00 b0

        Therefore the ith bit will be moved to 3i position, so moved by
        3i  - i = 2i bits. Therefore

        9th bit will move by 18 = 16 + 2 bits

        8th bit will move by 16 = 16 bits

        7th bit will move by 14 = 8 + 4 + 2 bits

        6th bit will move by 12 = 8 + 4 bits

        5th bit will move by 10 = 8 + 2 bits

        4th bit will move by 8 bits

        3rd bit will move by 6 = 4 + 2 bits

        2nd bit will move by 4 bits

        1st bit will move by 2 bits.

        0 bit will not move.

        We can move the bits with the same power of 2 moves together:

        9th and 8th bits will move 16 bits together.

        7th, 6th, 5th and 4th bits will move 8 bits together.

        7th, 6th, 3rd, and 2nd bits will move 4 bits together.

        9th, 7th, 5th, 3rd and 1st will move 2 bits together.
        """
        # Move 9th and 8th bits by 16 bits together.
        v = (v | v << 16) & 0b00000011000000000000000011111111
        # Move 7th, 6th, 5th, 4th bits by 8 bits together.
        v = (v | v << 8) & 0b00000011000000001111000000001111
        # Move 7th, 6th, 3rd, 2nd bits by 4 bits together.
        v = (v | v << 4) & 0b00000011000011000011000011000011
        # Move 9th, 7th, 5th, 3rd, 1st bits by 2 bits together.
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

    # Map the center coordinates of the bbx within range 0 ~ 2^10.
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
